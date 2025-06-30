from transformers import pipeline

# Lightweight model
summarizer = pipeline("summarization", model="knkarthick/MEETING_SUMMARY")

def summarize_text(text, max_chunk=1000):
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
    summary = ""
    for chunk in chunks:
        result = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
        summary += result[0]['summary_text'] + "\n\n"
    return summary
