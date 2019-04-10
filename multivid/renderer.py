from PIL import ImageFont, ImageDraw, Image
import numpy as np
import os

class ImageFontRenderer:
    def __init__(self, font_path):
        self.font_path = font_path
        self.font = ImageFont.truetype(font_path, 14, encoding="utf-8")
        assert (os.path.exists(self.font_path))
        self.base = 14

    def render(self, image, bbox, text):
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        base = self.base 
        while True:
            font = ImageFont.truetype(self.font_path, base, encoding="utf-8")
            w, h = font.getsize(text)
            if w > bbox.X - bbox.x or h > bbox.Y - bbox.y:
                base = base - 1
                break
            base = base + 1
        font = ImageFont.truetype(self.font_path, base, encoding="utf-8")
        draw.text((bbox.x, bbox.y), text, (0, 0, 0), font=font)
        image = np.array(image)
        return image

    def __call__(self, image, bbox, text):
        return self.render(image, bbox, text)

if __name__ == '__main__':
    pass

