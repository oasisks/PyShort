from transformers import BertTokenizer, BertForMaskedLM
import torch
import torch.nn.functional as F
from bert_score import score
from transformers import BertTokenizer, BertModel
import pandas as pd
import re
import random
import csv

# Load tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

# Load original BERT model
original_model = BertForMaskedLM.from_pretrained(model_name)

# Load fine-tuned model
fine_tuned_model_path = "./finetuned_bert/bert-mlm"  # Path where the fine-tuned model is saved
fine_tuned_model = BertForMaskedLM.from_pretrained(fine_tuned_model_path)

def predict_masked_token_with_probs(model, sentence, masked_token="[MASK]", top_k=1):
    """
    Predict the masked token in the sentence using the given model and return probabilities.
    """
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    masked_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits

    # Extract logits for the masked position and apply softmax for probabilities
    logits = predictions[0, masked_index, :].squeeze(0)
    probs = F.softmax(logits, dim=-1)
    # print(probs)

    all_tokens_with_probs = {
        tokenizer.decode([token_id]).strip(): prob.item() for token_id, prob in enumerate(probs)
    }

    # Get top-k predictions
    top_k_ids = torch.topk(probs, top_k).indices.tolist()
    top_k_probs = torch.topk(probs, top_k).values.tolist()

    # Decode tokens and pair with probabilities of the top-1 predictions
    try:
        predicted_tokens_with_probs = [
        (tokenizer.decode([token_id]).strip(), prob) for token_id, prob in zip(top_k_ids, top_k_probs)][0]
    except:
        predicted_tokens_with_probs = (None, 0)

    return all_tokens_with_probs, predicted_tokens_with_probs


def calculate_bertscore(candidate, reference, model_path):
        """
        Calculate BERTScore using a custom fine-tuned BERT model.

        Args:
            candidate (str): The candidate summary.
            reference (str): The reference summary.
            model_path (str): Path to the fine-tuned BERT model.

        Returns:
            Tuple containing precision, recall, and F1 scores.
        """
        # Calculate BERTScore
        P, R, F1 = score(
            [candidate],         # List of candidate summaries
            [reference],         # List of reference summaries
            model_type=model_path,  # Custom fine-tuned BERT model
            num_layers=12,        # Specify the layer if needed
            verbose=True
        )
        return P.item(), R.item(), F1.item()


def mask_text(text):
    """
    Given a text, randomly select a word to mask and replace it with [Mask].
    :param text: str, input text
    :return: a masked version of the text and the original word
    """
    # Tokenize the text into words
    words = re.findall(r'\b\w+\b', text)  # This regex matches words and excludes punctuation

    # Randomly select a word to mask
    word_to_mask = random.choice(words)

    # Replace the selected word with [Mask]
    masked_text = text.replace(word_to_mask, '[MASK]', 1)  # Only replace the first occurrence

    return masked_text, word_to_mask


def get_validation_accuracy():
    """
    Calculate the average probability of the correct word for the original and fine-tuned BERT models on a validation dataset.
    :return: Tuple containing the average probability of the correct word for the original and fine-tuned BERT models
    """
    df = pd.read_parquet("./finetuned_bert/articles.parquet", engine='pyarrow')

    validation_df = df.sample(frac=1/50, random_state=42)
    total_prob_fined_tuned = 0
    total_prob_original = 0

    total_correct_fined_tuned = 0
    total_correct_original = 0

    i = 0
    probabilities = [] # tuples of probabilities for original and fine-tuned correct prediction
    for index, row in validation_df.iterrows():

        masked_text, word = mask_text(row['text'])

        # Predict with original BERT
        original_predictions, correct_pred_original = predict_masked_token_with_probs(original_model, masked_text)
        correct_prob_original = original_predictions.get(word.lower(), 0)
        total_prob_original += correct_prob_original
        predicted_word_original = correct_pred_original[0]

        if predicted_word_original == word.lower():
            total_correct_original += 1

        # Predict with fine-tuned BERT
        fine_tuned_predictions, correct_pred_finetuned = predict_masked_token_with_probs(fine_tuned_model, masked_text)
        correct_prob_finetuned = fine_tuned_predictions.get(word.lower(), 0)
        total_prob_fined_tuned += correct_prob_finetuned
        predicted_word_finetuned = correct_pred_finetuned[0]

        if predicted_word_finetuned == word.lower():
            total_correct_fined_tuned += 1

        probabilities.append((correct_prob_original, correct_prob_finetuned))

        if i % 100 == 0:
            print(f"Processed {i}, {len(validation_df)}")
            print(total_prob_original, total_prob_fined_tuned, total_correct_original, total_correct_fined_tuned)
            print("#########")
        i+=1  

    return {
        "original_mean": total_prob_original/len(validation_df),
        "original_sum": total_prob_original,
        "fine_tuned_mean": total_prob_fined_tuned/len(validation_df),
        "fine_tuned_sum": total_prob_fined_tuned,
        "total_sample": len(validation_df),
        "original_correct": total_correct_original,
        "fine_tuned_correct": total_correct_fined_tuned,
        "original_accuracy": total_correct_original/len(validation_df),
        "fine_tuned_accuracy": total_correct_fined_tuned/len(validation_df),
        "probabilities": probabilities
    }
    

def write_to_csv(data, filename, fieldnames = ['original_correct_prob', 'finetuned_correct_prob']):
    """
    Write a list of dictionaries to a CSV file.
    @param data: list of dictionaries to write to the CSV file
    @param filename: str, name of the CSV file to write to

    @return: None
    """
    # Open the CSV file for writing
    with open(filename, mode='w', newline='', encoding='utf-8') as file:

        writer = csv.writer(file)
        writer.writerow(fieldnames)
        
        # Write each dictionary in the data list as a row
        for row in data:
            writer.writerow(row)
    

if __name__ == "__main__":

    # BERT for mlm

    # Example sentence
    # sentence = "The quick brown fox jumps over the [MASK] dog."
    # sentence = "WASHINGTON, May 1 (Reuters) - The U.S. Federal Aviation Administration on Tuesday said it was implementing new rules requiring additional inspections of fan blades in engines similar to the one that failed in a deadly Southwest Airlines accident last month in Pennsylvania. The new rules follow an emergency directive issued last month by the FAA and European regulators requiring the inspection of nearly 700 engines CFM56-7B worldwide within 20 days on Boeing 737 NG airplanes. The FAA said another group of CFM56-7B engines will need to be inspected by August under the regulation. The FAA also said it is requiring repeat inspections of engines as part of Tuesday’s action. The announcement comes the same day President Donald Trump is set to meet with passengers and the flight crew at the White House of [MASK] Flight 1380. (Reporting by David Shepardson Editing by Chizu Nomiyama)"

    # # Predict with original BERT
    # print("Original BERT predictions:")
    # original_predictions = predict_masked_token_with_probs(original_model, sentence)
    # for token, prob in original_predictions:
    #     print(f"{token}: {prob:.4f}")

    # # Predict with fine-tuned BERT
    # print("\nFine-tuned BERT predictions:")
    # fine_tuned_predictions = predict_masked_token_with_probs(fine_tuned_model, sentence)
    # for token, prob in fine_tuned_predictions:
    #     print(f"{token}: {prob:.4f}")

    # print("#########")

    results = get_validation_accuracy()
    for k,v in results.items():
        if k == "probabilities":
            continue
        print(f"{k}: {v}")

    write_to_csv(results["probabilities"], "validation_probabilities.csv")

    ###############################################

    # BERT for BERTScore

    # Paths to fine-tuned BERT model and tokenizer
    # fine_tuned_model_path = "./bert-mlm"
    # original_model_name = "bert-base-uncased"

    # reference = "WASHINGTON, May 1 (Reuters) - The U.S. Federal Aviation Administration on Tuesday said it was implementing new rules requiring additional inspections of fan blades in engines similar to the one that failed in a deadly Southwest Airlines accident last month in Pennsylvania. The new rules follow an emergency directive issued last month by the FAA and European regulators requiring the inspection of nearly 700 engines CFM56-7B worldwide within 20 days on Boeing 737 NG airplanes. The FAA said another group of CFM56-7B engines will need to be inspected by August under the regulation. The FAA also said it is requiring repeat inspections of engines as part of Tuesday’s action. The announcement comes the same day President Donald Trump is set to meet with passengers and the flight crew at the White House of Southwest Flight 1380. (Reporting by David Shepardson Editing by Chizu Nomiyama)"    

    # candidate = "The U.S. Federal Aviation Administration has announced new rules for inspecting fan blades in engines like the one that failed in a Southwest Airlines accident last month. The new rules require additional inspections of CFM56-7B engines on Boeing 737 NG airplanes, following an emergency directive issued last month. The FAA is also requiring repeat inspections of engines as part of the new regulation. The announcement coincides with President Donald Trump's meeting with passengers and crew from Southwest Flight 1380 at the White House."

    # # Compute BERTScore
    # precision, recall, f1_score = calculate_bertscore(candidate, reference, fine_tuned_model_path)
    # print(f"BERTScore using fine-tuned BERT model:")
    # print(f"precision: {precision:.4f}")
    # print(f"recall: {recall:.4f}")
    # print(f"F1 Score: {f1_score:.4f}")

    # # Compute BERTScore using original BERT model
    # precision, recall, f1_score = calculate_bertscore(candidate, reference, original_model_name)
    # print(f"BERTScore using original BERT model:")
    # print(f"precision: {precision:.4f}")
    # print(f"recall: {recall:.4f}")
    # print(f"F1 Score: {f1_score:.4f}")