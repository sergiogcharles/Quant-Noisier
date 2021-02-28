import argparse

from fairseq.models.roberta import RobertaModel

def main(args):
    roberta = RobertaModel.from_pretrained(
        args.checkpoint_dir,
        args.checkpoint_file,
        data_name_or_path='RTE-bin'
    )

    label_fn = lambda label: roberta.task.label_dictionary.string(
        [label + roberta.task.label_dictionary.nspecial]
    )
    ncorrect, nsamples = 0, 0
    roberta.cuda()
    roberta.eval()
    with open('RTE/dev.tsv') as fin:
        fin.readline()
        for index, line in enumerate(fin):
            tokens = line.strip().split('\t')
            sent1, sent2, target = tokens[1], tokens[2], tokens[3]
            tokens = roberta.encode(sent1, sent2)
            prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
            prediction_label = label_fn(prediction)
            ncorrect += int(prediction_label == target)
            nsamples += 1
    print('| Accuracy: ', float(ncorrect)/float(nsamples))

if __name__ == '__main__':
    #TODO: Make this arg parsing cleaner
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir")
    parser.add_argument("checkpoint_file")
    args = parser.parse_args()
    main(args)
