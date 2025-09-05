# Fine-tuning SLM dataset

We developed a specialized fine-tuning dataset for small language models to enhance high-level task configuration in underwater robotics. Through fine-tuning on this dataset, we significantly improved the parsing accuracy of small models for underwater robot task specifications. Experimental validation was conducted across four different small language models.


<p align="left">
<img src="/figure/ocean.png" alt="Additional Image 1" width="300" height="300"/>
</p>

# Prerequisites and requirement 
* GPU: At least 32GB VRAM
* Install [ROS2 Humble](https://docs.ros.org/en/humble/)
* Ubuntu 20.04/22.04
* Python 3.9+


# Fine tuning deepseek 8B model
* Training dataset: /datasets/generated_trajectory_data2.json
* Test dataset： /datasets/test_questions2.csv

First, you should refer to your own file path and modify the model's input path accordingly.
```
model_name = "your_model_path"
...
df = pd.read_json('/your_path/generated_trajectory_data2.json')
...
trainer.save_model('your_path')
```

Then, execute the fine-tuning program code.
```
python Fine_8B.py
```

# Evaluate the accuracy

Additionally, you should refer to your own file paths and modify the input paths for the models in the evaluation file accordingly.
```
model_name = "your_model_path"
...
model = AutoModelForCausalLM.from_pretrained(
    "your_path/Llama_8b_LoRA",
    device_map="auto",
    trust_remote_code=True
)
...
test_df = pd.read_csv("your_path/test_questions2.csv")
...
trainer.save_model('your_path')
```

Then, execute the test evaluation program code.
```
python Fine_8B_acc.py
```


# Chat with fined model with ROS2
This file is responsible for invoking the fine-tuned model and converting model outputs into standard ROS2 Topic message format through the ROS2 mechanism.
```
python chat_ros2.py
```
Then, the chat interface appears as follows:
<p align="left">
<img src="/figure/chat_interface.png" alt="Additional Image 2" width="500" height="500"/>
</p>

![Publication](https://img.shields.io/badge/Publication-Coming%20Soon-blue)
