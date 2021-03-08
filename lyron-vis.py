import argparse

from fairseq.models.roberta import RobertaModel

import plotly
import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA

from tqdm import tqdm

def main():
    roberta_unquantized = RobertaModel.from_pretrained("checkpoints/roberta/rte-no-quant-noise", "checkpoint_best.pt", data_name_or_path='RTE-bin')
    roberta_quantized = RobertaModel.from_pretrained("checkpoints/roberta/rte-scalar-4-quant-noise", "checkpoint_best.pt", data_name_or_path='RTE-bin')

    unquantized_label_fn = lambda label: roberta_unquantized.task.label_dictionary.string(
        [label + roberta_unquantized.task.label_dictionary.nspecial]
    )

    quantized_label_fn = lambda label: roberta_quantized.task.label_dictionary.string(
        [label + roberta_quantized.task.label_dictionary.nspecial]
    )

    roberta_unquantized.eval()
    roberta_quantized.eval()
    results = []
    with open('RTE/dev.tsv') as fin:
        fin.readline()
        for index, line in tqdm(enumerate(fin)):
            #TODO: Implement the vis piece and eventually clean this up (should have some more decomposition)
            tokens = line.strip().split('\t')
            sent1, sent2, target = tokens[1], tokens[2], tokens[3]
            tokens_unquantized = roberta_unquantized.encode(sent1, sent2)
            unquantized_embedding = roberta_unquantized.extract_features(tokens_unquantized, return_all_hiddens=True)[0]
            tokens_quantized = roberta_quantized.encode(sent1, sent2)
            quantized_embedding = roberta_quantized.extract_features(tokens_quantized, return_all_hiddens=True)[0]
            assert all(tokens_unquantized == tokens_quantized)
            prediction_unquantized = roberta_unquantized.predict('sentence_classification_head', tokens_unquantized).argmax().item()
            prediction_quantized = roberta_quantized.predict('sentence_classification_head', tokens_quantized).argmax().item()
            prediction_label_unquantized = unquantized_label_fn(prediction_unquantized)
            prediction_label_quantized = quantized_label_fn(prediction_quantized)
            if prediction_label_unquantized == target and prediction_label_quantized == target:
                color = "purple"
            elif prediction_label_unquantized != target and prediction_label_quantized == target:
                color = "blue"
            elif prediction_label_unquantized == target and prediction_label_quantized != target:
                color = "red"
            else:
                color = "gray"
            results.append((quantized_embedding - unquantized_embedding, color))
            
    word_vectors = np.array([np.mean(res[0].cpu().detach().numpy(), axis=1).reshape(-1,) for res in results])

    two_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:2]
    data = []
    count = 0

    for i in range (len(results)):

        trace = go.Scatter(x = two_dim[i:i+1,0], y = two_dim[i:i+1,1],marker = {'size': 10, 'opacity': 0.8, 'color': results[i][1]})

        data.append(trace)

    # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable.  Also, instead of using
    # variable three_dim, use the variable that we have declared earlier (e.g two_dim)

    #data.append(trace_input)

# Configure the layout

    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
        x=1,
        y=0.5,
        font=dict(
            family="Courier New",
            size=25,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 15),
        autosize = False,
        width = 1000,
        height = 1000
        )


    plot_figure = go.Figure(data = data, layout = layout)
    breakpoint()
    plot_figure.show()

if __name__ == '__main__':
    #TODO: Make this arg parsing cleaner
    """
    #Commenting this out for now
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir")
    parser.add_argument("checkpoint_file")
    args = parser.parse_args()
    """
    main()
