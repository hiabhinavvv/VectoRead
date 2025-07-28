from transformers import CLIPProcessor, CLIPModel
from PIL import Image



"""
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
image = Image.open("data/image.jpeg")
text_classes = ["america", "japan"]
inputs = processor(text=text_classes,images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

predicted_class = text_classes[probs.argmax()]
print(predicted_class, "| probability = ", round(float(probs[0][probs.argmax()]), 4))
"""

