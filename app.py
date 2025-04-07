import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from unsloth import FastLanguageModel

# ---------------------------
# Load Classification Model
# ---------------------------
clf_model_id = "XoXonn/ClassificationModel-s-nlp_roberta_toxicity_classifier"
clf_tokenizer = AutoTokenizer.from_pretrained(clf_model_id)
clf_model = AutoModelForSequenceClassification.from_pretrained(clf_model_id).eval()

# Classification label mapping (based on your setup)
labels = ["Insult", "Threat", "Sexual Harassment", "Non-Toxic"]

# ---------------------------
# Load Detoxification Model
# ---------------------------
detox_model_id = "XoXonn/DetoxModel-unsloth_Meta-Llama-3.1-8B"
detox_model, detox_tokenizer = FastLanguageModel.from_pretrained(
    model_name=detox_model_id,
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(detox_model)

# ---------------------------
# Alpaca Prompt
# ---------------------------
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Task: Detoxify the text without changing its original meaning or personal references. Replace offensive or inappropriate words with neutral or more respectful alternatives while keeping the overall context, structure, and tone intact.

- Preserve references to people such as "you", "he", "she", "they", "this person", etc. — do not replace or remove them.

- Maintain the input’s language style (e.g., casual, formal, slang) and tone while detoxifying the content.

- For repeated or stylized offensive terms (e.g., "nigga nigga nigga"), do not default to generic or meaningless repetition (e.g., "person person person"). Instead, rewrite it in a contextually meaningful and stylistically consistent way that reflects the original tone without being offensive.

- Ensure the detoxified version communicates the same emotion or message, just without using harmful language.

### Example Transformations:
Original: "you are fucking retarded"
Detoxified: "you lack intelligence"

Original: "ur mom's a whore"
Detoxified: "your mom engages in questionable behavior"

Original: "this idiot is so damn annoying"
Detoxified: "this person is extremely irritating"

Now, detoxify the following text while maintaining its context, personal references, and language style:

### Input:
{}

### Response:
{}
"""

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Toxic Comment Detoxifier", layout="centered")
st.title("🧼 Toxic Comment Classifier + Detoxifier")
user_input = st.text_area("Enter a comment to analyze:")

if st.button("Run"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Classifying..."):
            tokens = clf_tokenizer(user_input, return_tensors="pt")
            with torch.no_grad():
                outputs = clf_model(**tokens)
                predicted = torch.argmax(outputs.logits, dim=1).item()
                label = labels[predicted]

        st.markdown(f"**Classification Result:** `{label}`")

        if label == "Non Toxic":
            st.success("This comment is not toxic. ✅")
        else:
            with st.spinner("Detoxifying..."):
                full_prompt = alpaca_prompt.format(user_input, "")
                input_tokens = detox_tokenizer([full_prompt], return_tensors="pt").to("cuda")
                output = detox_model.generate(**input_tokens, max_new_tokens=64, use_cache=False)
                decoded = detox_tokenizer.batch_decode(output, skip_special_tokens=False)[0]

                if "### Response:" in decoded:
                    detoxified = decoded.split("### Response:")[1].split("<|end_of_text|>")[0].strip()
                else:
                    detoxified = decoded.strip()

            st.markdown("### 🧽 Detoxified Comment")
            st.info(detoxified)
