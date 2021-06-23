import tarfile
import os

root = "./tmp_model"
model = "distilbert-base-uncased-finetuned-sst-2-english"

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(model)

model = AutoModelForSequenceClassification.from_pretrained(model)

model.save_pretrained(root)
tokenizer.save_pretrained(root)


def main():
    cwd = os.getcwd()
    os.chdir(root)
    files = [os.path.join(path, name) for path, _, files in os.walk(".") for name in files]
    with tarfile.open(os.path.join(cwd, "model.tar.gz"), "w:gz") as tar:
        for file in files:
            tar.add(file)


if __name__ == "__main__":
    main()
