"""exports image data in binary format embedded in JSON so we can use it in Cloud ML
"""
import base64
import io
import json
from PIL import Image
from predict import IMAGE_URLS
from predict import load_images

def convert_img_to_json_bytes():
    predict_instance_json = "svhn/inputs.json"
    with open(predict_instance_json, "wb") as fp:
        for image in IMAGE_URLS:
            img = Image.open(image)
            # img = img.resize((width, height), Image.ANTIALIAS)
            output_str = io.BytesIO()
            img.save(output_str, "JPEG")
            fp.write(
                # json.dumps({"image_bytes": {"b64": base64.b64encode(output_str.getvalue())}}) + "\n")
                json.dumps({"image_bytes": {"b64": base64.b64encode(output_str.getvalue())}}))
            output_str.close()

## Run script
if __name__ == "__main__":
    # imgs = load_images()
    # print imgs[0].shape
    # convert_img_to_json_bytes()


# Read https://stackoverflow.com/questions/46216095/using-gcloud-ml-serving-for-large-images?rq=1 as contains good info on Tensor input format
# Example of converting img to JSON encoded
# ---> https://github.com/GoogleCloudPlatform/cloudml-samples/blob/master/flowers/images_to_json.py
