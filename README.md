# Fine-tuning SLM dataset

We developed a specialized fine-tuning dataset for small language models to enhance high-level task configuration in underwater robotics. Through fine-tuning on this dataset, we significantly improved the parsing accuracy of small models for underwater robot task specifications. Experimental validation was conducted across four different small language models.


<p align="left">
<img src="ocean.png" alt="Additional Image 1" width="300" height="300"/>
</p>

# Prerequisites and requirement 
* GPU: At least 32GB VRAM
* Install [ROS2 Humble](https://docs.ros.org/en/humble/)
* Ubuntu 20.04/22.04
* Python 3.9+


# Fine tuning deepseek 8B model
You should refer to your own file path and modify the model's input path accordingly.
```
python Fine_8B.py
```

# Evaluate the accuracy
Also, you should refer to your own file path and modify the model's input path accordingly.
```
python Fine_8B_acc.py
```


# Chat with fined model
This file is responsible for invoking the fine-tuned model and converting model outputs into standard ROS2 Topic message format through the ROS2 mechanism.
```
python chat_ros2.py
```

![Publication](https://img.shields.io/badge/Publication-Coming%20Soon-blue)
