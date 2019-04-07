from PIL import ImageFont, ImageDraw, Image

class ImageFontRenderer:
    def __init__(self, font_path):
        self.font = ImageFont.truetype(font_path, 14, encoding="utf-8")

    def render(self, image, bbox, text):
        draw = ImageDraw.Draw(image)
        draw.text((bbox.x, bbox.y), text, (0, 0, 0), font=self.font)
        return image
