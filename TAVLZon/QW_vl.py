import os
from openai import OpenAI
import base64
import ast

text3  ="You are a helpful assistant.\
Your answer should be formatted, and you can only answer yes or no in lowercase.\
 "

Text_1 = "You are a wheeled mobile robot working in an indoor environment. \
Your task is to find the location of the specified object in the room. \
I will provide you with a specified object, and you need to answer with the category of room that has the strongest association with that object. \
If the object does not have a single room category with the strongest association, or if it is strongly associated with multiple rooms, then you only need to respond: 'Weak association' — there is no need to specify a room type.\
For example: \
(1) Specified object: bed , you can respond: Strong association, bedroom. \
(2) Specified object: toilet , you can respond: Strong association, bathroom. \
(3)If the specified object is a potted plant, it could appear in any room, so you only need to respond: Weak association. \
Your answer should be formatted as a dict, forexample: {'association':'Strong association','room':bathroom}.\ "


text_vl  ="You are a helpful assistant.\
I will provide you with an indoor panoramic image composed of scenes from six directions. I need you to determine the type of the room based on the image. \
Your answer should be formatted, and you can only answer yes or no in lowercase.\
 "

def QW_VL(image_path,text_1,text_2):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="qwen-vl-max-latest",
        messages=[
            {
                "role": "system",
                "content": [{"type": "text", "text": text_1}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                    {"type": "text", "text": text_2},
                ],
            },
        ],
    )

    return completion.choices[0].message.content



def QW_LL(text_1,text_2):

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus",
        messages=[
            {"role": "system", "content": text_1},
            {"role": "user", "content": text_2},
        ],
    )

    answer_dict = ast.literal_eval(completion.choices[0].message.content)

    return answer_dict
