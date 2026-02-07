import os
import sys
import torch

# Add the project root to sys.path to allow importing from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models import load_t5_vietnamese_model

def summarize(text, tokenizer, model, max_length=150, num_beams=3):
    """
    Generates a summary for the given text using the T5 model.
    """
    # T5 models often expect a prefix for the task
    inputs = tokenizer("Tóm tắt tin tức:" + text, return_tensors="pt", max_length=512, truncation=True)
    
    # Move tensors to the same device as the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=48,
            no_repeat_ngram_size=3,
            num_beams=num_beams,
            length_penalty=2.2,
            early_stopping=True
        )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main():
    print("Loading model and tokenizer...")
    tokenizer, model, device = load_t5_vietnamese_model()
    
    # Sample Vietnamese text (a news snippet)
    sample_text = """
    Ok mọi người bắt đầu họp nhé. Hôm nay mình họp nhanh để cập nhật tiến độ dự án OCR chấm bài tự luận và bàn thêm về hướng demo cho nhà đầu tư sắp tới. Thời gian không nhiều nên mình đi thẳng vào vấn đề luôn. Về tiến độ kỹ thuật trước. Hiện tại phần mô hình chấm điểm đã fine-tune xong phiên bản đầu tiên cho tiếng Việt. Dataset dùng khoảng hơn 20 nghìn bài tự luận, chủ yếu là các môn xã hội. Kết quả ban đầu khá ổn, model đã hiểu được ý chính và cho điểm tương đối sát với giảng viên. Tuy nhiên vẫn còn lỗi ở những bài trả lời quá ngắn hoặc viết lan man, không đi đúng trọng tâm. Phần này đang được xử lý bằng cách bổ sung thêm dữ liệu và chỉnh lại tiêu chí chấm điểm.
    """
    
    print("Original Text:")
    print(sample_text.strip())
    
    print("Generating summary...")
    summary = summarize(sample_text, tokenizer, model)
    
    print("Summary:")
    print(summary)

if __name__ == "__main__":
    main()
