import torch
from PIL import Image
from transformers import XLMRobertaTokenizer

class ModelInference:
    def __init__(self, model_path, tokenizer_path, valid_labels, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化模型和分词器
        :param model_path: 训练好的模型路径
        :param tokenizer_path: XLM-Roberta 分词器路径
        :param device: 运行设备（CPU 或 GPU）
        """
        self.device = device
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        self.model.to(self.device)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_path)
        self.valid_labels = valid_labels

    def preprocess_image(self, image_path):
        """
        图像预处理
        :param image_path: 图像路径
        :return: 预处理后的图像张量
        """
        img = Image.open(image_path).convert("RGB")
        transform = self.model.img_transform  # 使用模型内定义的图像转换方法
        return transform(img).unsqueeze(0).to(self.device)

    def predict(self, input_text, image_path=None, valid_labels=None, threshold=0.7, default_label="UNKNOWN"):
        """
        进行推断
        :param input_text: 输入文本
        :param image_path: 可选，输入图像路径
        :param valid_labels: 可选，合法标签列表
        :param threshold: 置信度阈值
        :param default_label: 默认标签
        :return: 推断结果（标签、置信度）
        """
        # 1. 格式化文本输入
        prompt = f"[LABEL] {input_text}"
        text_inputs = self.tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(self.device)

        # 2. 处理图像输入
        if image_path:
            image_inputs = self.preprocess_image(image_path)
            logits = self.model.forward(text_inputs["input_ids"], images=[image_inputs])
        else:
            logits = self.model.forward(text_inputs["input_ids"], images=None)

        logits = logits.squeeze(0)

        # 3. 解码输出
        output_idx = logits.argmax(dim=-1).item()
        confidence = torch.softmax(logits, dim=-1).max().item()
        predicted_label = self.model.map_to_valid_label(output_idx, valid_labels) if valid_labels else f"Label_{output_idx}"

        # 4. 检查置信度
        if confidence < threshold or (valid_labels and predicted_label not in valid_labels):
            predicted_label = default_label

        return {
            "label": predicted_label,
            "confidence": confidence,
        }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model Inference Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer")
    parser.add_argument("--input_text", type=str, required=True, help="Input text for prediction")
    parser.add_argument("--image_path", type=str, help="Optional image path for prediction")
    parser.add_argument("--valid_labels", type=str, nargs="*", help="Optional list of valid labels")
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold")
    parser.add_argument("--default_label", type=str, default="UNKNOWN", help="Default label if confidence is low")

    args = parser.parse_args()

    # 初始化推断类
    inference = ModelInference(args.model_path, args.tokenizer_path)

    # 进行推断
    result = inference.predict(
        input_text=args.input_text,
        image_path=args.image_path,
        valid_labels=args.valid_labels,
        threshold=args.threshold,
        default_label=args.default_label,
    )

    print("Prediction Result:", result)
