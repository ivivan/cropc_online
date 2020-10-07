import streamlit as st
import altair as alt
import operator
import numpy as np
import pandas as pd
import io
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from catalyst.utils import set_global_seed
from catalyst.dl import SupervisedRunner, Runner
from catalyst.utils import metrics
from catalyst.dl.callbacks import AccuracyCallback, AUCCallback, F1ScoreCallback, EarlyStoppingCallback, CriterionCallback

import torch
import torch.nn as nn
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from torch.autograd import Variable
st.set_option('deprecation.showfileUploaderEncoding', False)


SEED = 15
set_global_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionModel(torch.nn.Module):
    def __init__(self, batch_size, input_dim, hidden_dim, output_dim, recurrent_layers, dropout_p):
        super(AttentionModel, self).__init__()

        self.batch_size = batch_size
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.recurrent_layers = recurrent_layers
        self.dropout_p = dropout_p

        self.input_embeded = nn.Linear(input_dim, hidden_dim//2)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(input_size=hidden_dim//2, hidden_size=hidden_dim, num_layers=recurrent_layers,
                            bidirectional=True)

        self.self_attention = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.ReLU(True),
            nn.Linear(hidden_dim*2, 1)
        )

        # initialize LSTM forget gate bias to be 1 as recommanded by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        self.output_linear = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.label = nn.Linear(hidden_dim*4, output_dim)

    def forward(self, input_sentences, batch_size=None):

        input = self.dropout(torch.tanh(self.input_embeded(input_sentences)))
        input = input.permute(1, 0, 2)
        if batch_size is None:
            h_0 = Variable(torch.zeros(2*self.recurrent_layers,
                                       self.batch_size, self.hidden_dim).to(device))
            c_0 = Variable(torch.zeros(2*self.recurrent_layers,
                                       self.batch_size, self.hidden_dim).to(device))
        else:
            h_0 = Variable(torch.zeros(2*self.recurrent_layers,
                                       batch_size, self.hidden_dim).to(device))
            c_0 = Variable(torch.zeros(2*self.recurrent_layers,
                                       batch_size, self.hidden_dim).to(device))

        output, (final_hidden_state, final_cell_state) = self.lstm(
            input, (h_0, c_0))
        output = output.permute(1, 0, 2)

        attn_ene = self.self_attention(output)

        attn_ene = attn_ene.view(
            self.batch_size, -1)

        # mannual masking, force model focus on previous time index
        mask_one = torch.ones(
            size=(self.batch_size, attn_ene.shape[1]), dtype=torch.long).to(device)
        mask_zero = torch.zeros(size=(self.batch_size, 6),
                                dtype=torch.long).to(device)
        mask_one[:, -6:] = mask_zero
        attn_ene = attn_ene.masked_fill(mask_one == 0, -np.inf)

        attns = F.softmax(attn_ene, dim=1).unsqueeze(2)

        final_inputs = (output * attns).sum(dim=1)
        final_inputs2 = output.sum(dim=1)

        combined_inputs = torch.cat([final_inputs, final_inputs2], dim=1)

        logits = self.label(combined_inputs)

        return logits, attns
        # return {"logits": logits, "attention": attention_scores}


class CustomRunner(Runner):

    def predict_batch(self, batch):                 # here is the trick
        return self.model(batch[0].to(self.device))

    def _handle_batch(self, batch):
        x, y = batch
        # y_hat, attention = self.model(x)
        outputs, _ = self.model(x)

        loss = F.cross_entropy(outputs['logits'], y)
        accuracy01, accuracy02 = metrics.accuracy(
            outputs['logits'], y, topk=(1, 2))
        self.batch_metrics = {
            "loss": loss,
            "accuracy01": accuracy01,
            "accuracy02": accuracy02,
        }

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')  # don't have GPU
    return device


def numpy_to_tensor(ay, tp):
    device = get_device()
    return torch.from_numpy(ay).type(tp).to(device)


def play_line_plots(df):
    df_temp = pd.DataFrame(df.values, columns=['NDVI'])
    base = alt.Chart(df_temp.reset_index()).mark_line().encode(
        x=alt.X('index', axis=alt.Axis(title='Day')),
        y=alt.Y('NDVI', axis=alt.Axis(title='NDVI'))).properties(
        width=600,
        height=400).interactive()
    st.altair_chart(base)


def play_bar_plots(df):
    df_temp = pd.DataFrame(df, columns=['Attention'])

    base = alt.Chart(df_temp.reset_index()).mark_bar().encode(
        x=alt.X('index', axis=alt.Axis(title='Day')),
        y=alt.Y('Attention', axis=alt.Axis(title='Attention'))).properties(
        width=600,
        height=400).interactive()
    st.altair_chart(base)


@st.cache
def read_scaler(path):
    scaler_data_ = np.load(path)
    return scaler_data_


def prepare_input(df, scalerinfo):
    # normalizeation
    scaler = StandardScaler()
    scaler.fit(df)
    scaler.scale_, scaler.mean_, scaler.var_ = scalerinfo[0], scalerinfo[1], scalerinfo[2]
    X_test = scaler.transform(df)

    # prepare PyTorch Datasets
    X_test_tensor = numpy_to_tensor(X_test, torch.FloatTensor)
    X_test_tensor = torch.unsqueeze(X_test_tensor, 2)
    valid_ds = TensorDataset(X_test_tensor)
    # #### model inference
    BATCH_SIZE = 1
    test_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE,
                         shuffle=False, drop_last=True, num_workers=0)

    return test_dl


@st.cache
def prepare_model():

    INPUT_DIM = 1
    OUTPUT_DIM = 5
    HID_DIM = 40
    DROPOUT = 0.3
    RECURRENT_Layers = 2
    BATCH_SIZE = 1
    num_classes = 5

    # model, criterion, optimizer, scheduler
    model = AttentionModel(BATCH_SIZE, INPUT_DIM, HID_DIM,
                           OUTPUT_DIM, RECURRENT_Layers, DROPOUT).to(device)

    return model


def execute_model(model, input_data):
    # # model training
    runner = CustomRunner()
    class_names = ['Barley', 'Canola', 'Chick Pea', 'Lentils', 'Wheat']
    predictions = np.vstack(list(map(
        lambda x: x[0].cpu().numpy(),
        runner.predict_loader(model=model,
                              loader=input_data, resume="./best_full.pth")
    )))

    attentions = np.vstack(list(map(
        lambda x: x[1].cpu().numpy(),
        runner.predict_loader(model=model,
                              loader=input_data, resume="./best_full.pth")
    )))
    attentions = attentions.reshape(attentions.shape[1], 1)

    probabilities = []
    pred_labels = []
    true_labels = []
    pred_classes = []
    true_classes = []

    for i, logits in enumerate(predictions):
        probability = torch.softmax(torch.from_numpy(logits), dim=0)
        pred_label = probability.argmax().item()
        probabilities.append(probability.cpu().numpy())
        pred_labels.append(pred_label)
        pred_classes.append(class_names[pred_label])

    probabilities_df = pd.DataFrame(probabilities)
    pred_labels_df = pd.DataFrame(pred_labels)
    pred_classes_df = pd.DataFrame(pred_classes)

    results = pd.concat([probabilities_df,
                         pred_classes_df], axis=1)
    results.columns = ['Prob_Barley', 'Prob_Canola', 'Prob_Chick_Pea', 'Prob_Lentils',
                       'Prob_Wheat', 'Pred_class']

    return probabilities, pred_classes, pred_labels, results, attentions


def check_prediction(pred_classes, pred_label, ndvi_nrow, groud_truth_df):
    class_names = ['Barley', 'Canola', 'Chick Pea', 'Lentils', 'Wheat']
    ground_truth = groud_truth_df.iloc[ndvi_nrow].to_numpy()
    st.markdown("Ground Truth:{}".format(class_names[ground_truth[0]]))
    if np.equal(ground_truth, pred_label):
        st.markdown("Correct!!!")
        st.balloons()


if __name__ == '__main__':
    st.sidebar.header("Crop Classification Demo")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Using Steps")
    st.sidebar.markdown("---")
    st.sidebar.markdown("1. Upload NDVI Data")
    st.sidebar.markdown("2. NDVI Visualization")
    st.sidebar.markdown("3. Crop Classification")
    st.sidebar.markdown("---")
    st.sidebar.subheader("LSTM with Self Attention")
    st.sidebar.markdown("---")
    # model_structure = st.sidebar.button("Show Model Structure")

    st.header('Crop Classification Demo')
    st.subheader("Upload Crop NDVI Data")
    k = st.number_input("Maximum No. of Rows to Read", min_value=10,
                        max_value=1000, step=1, value=10, key='readinput')
    results = None
    uploaded_file = st.file_uploader(
        "Choose a CSV file (Maximum 1000 Rows for Performance)", type="csv", key='test')
    st.subheader("Upload Ground Truth Label (Only for Testing)")
    ground_truth_file = st.file_uploader(
        "Choose a CSV file (Should Match the NDVI CSV)", type="csv", key='truth')
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, nrows=k)
        st.write(data)
        st.subheader("Curve Visualization")
        st.line_chart(data.T.to_numpy())
        max_row = data.shape[0]-1
        st.subheader("Plot single NDVI curve")
        ndvi_nrow = st.number_input(
            "Pick up a row", min_value=0, max_value=max_row, step=1, value=0, key='singleinput')
        picked_ndvi = data.iloc[ndvi_nrow]
        show_ndvi = st.button("Show single NDVI Curve")
        if show_ndvi:
            play_line_plots(picked_ndvi)
        st.subheader("Crop Classification")
        run_model = st.button("Run ML model")
        if run_model:
            with st.spinner('Model Running, Input Curve Row No.{}'.format(ndvi_nrow)):
                picked_input = data.iloc[[ndvi_nrow]]
                scaler_info = read_scaler('./standard_scaler.npy')
                model_input = prepare_input(picked_input, scaler_info)
                model_instance = prepare_model()
                probabilities, pred_classes, pred_labels, results, attentions = execute_model(
                    model_instance, model_input)
                st.success('Finish. Input Curve Row No.{}'.format(ndvi_nrow))
        st.subheader("Results")
        if results is not None:
            st.write(results.style.highlight_max(axis=1))
            # play_bar_plots(attentions)
            if ground_truth_file is not None:
                data = pd.read_csv(ground_truth_file)
                check_prediction(pred_classes, pred_labels, ndvi_nrow, data)
