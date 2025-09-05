from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
import torch
import json
from datetime import datetime
import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading

# Server Equipment Selection
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


class TrajectoryAssistant(Node):
  def __init__(self, base_model_path, lora_model_path):
      super().__init__('trajectory_assistant')
      
      self.base_model_path = base_model_path
      self.lora_model_path = lora_model_path
      self.model = None
      self.tokenizer = None
      self.conversation_history = []
      
      # creating ROS2 Publishers and Subscribers
      self.response_publisher = self.create_publisher(String, 'assistant_response', 10)
      self.query_subscriber = self.create_subscription(
          String,
          'assistant_query',
          self.query_callback,
          10
      )
      
      self.get_logger().info('Trajectory Assistant ROS2 Node initialized')
  
  def clean_lora_config(self, config_path):
      """Backup Your Configuration Files"""
      try:
          with open(config_path, 'r', encoding='utf-8') as f:
              config = json.load(f)
          
          # definition of standard LoRA configuration parameters
          standard_lora_keys = {
              'peft_type', 'auto_mapping', 'base_model_name_or_path', 
              'revision', 'task_type', 'inference_mode', 'r', 'lora_alpha', 
              'lora_dropout', 'target_modules', 'fan_in_fan_out', 'bias',
              'modules_to_save', 'init_lora_weights', 'layers_to_transform',
              'layers_pattern', 'rank_pattern', 'alpha_pattern'
          }
          
          original_config = config.copy()
          cleaned_config = {}
          removed_keys = []
          
          for key, value in config.items():
              if key in standard_lora_keys:
                  cleaned_config[key] = value
              else:
                  removed_keys.append(key)
                    
          # Ensure the necessary parameters
          if 'peft_type' not in cleaned_config:
              cleaned_config['peft_type'] = 'LORA'
          if 'task_type' not in cleaned_config:
              cleaned_config['task_type'] = 'CAUSAL_LM'
          
          # Backup Creation
          backup_path = config_path + '.backup'
          if not os.path.exists(backup_path):
              with open(backup_path, 'w', encoding='utf-8') as f:
                  json.dump(original_config, f, indent=2)
              self.get_logger().info(f"Original config backed up to: {backup_path}")
          
          # Saving the cleaned configuration 
          with open(config_path, 'w', encoding='utf-8') as f:
              json.dump(cleaned_config, f, indent=2)
          
          return cleaned_config
          
          
      except Exception as e:
          self.get_logger().error(f"Error cleaning config: {e}")
          return None
  
  def load_model(self):
      """load the setting model"""
      try:
          self.get_logger().info("Loading tokenizer...")
          self.tokenizer = AutoTokenizer.from_pretrained(
              self.base_model_path, 
              trust_remote_code=True
          )
          self.tokenizer.pad_token = self.tokenizer.eos_token
          self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
          self.tokenizer.padding_side = "right"
          
          self.get_logger().info("Loading base model...")
          base_model = AutoModelForCausalLM.from_pretrained(
              self.base_model_path,
              torch_dtype=torch.float16,
              device_map={"": 0},
              trust_remote_code=True
          )
          
          # Checking and Cleaning Configuration Files
          config_path = os.path.join(self.lora_model_path, 'adapter_config.json')
          self.get_logger().info(f"Processing config file: {config_path}")
          
          if os.path.exists(config_path):
              # Cleaning Configuration Files totally
              cleaned_config = self.clean_lora_config(config_path)
              
              if cleaned_config:
                  self.get_logger().info("Attempting to load LoRA with cleaned config...")
                  try:
                      self.model = PeftModel.from_pretrained(base_model, self.lora_model_path)
                      self.get_logger().info("✅ LoRA model loaded successfully!")
                      return
                  except Exception as e:
                      self.get_logger().error(f"LoRA loading failed even with cleaned config: {e}")
              
              # If failed, setting by self
              self.get_logger().info("Attempting to rebuild config from scratch...")
              if self.rebuild_lora_config(config_path):
                  try:
                      self.model = PeftModel.from_pretrained(base_model, self.lora_model_path)
                      self.get_logger().info("✅ LoRA model loaded with rebuilt config!")
                      return
                  except Exception as e:
                      self.get_logger().error(f"Failed with rebuilt config: {e}")
          
          # Last choice, only using the based model
          self.get_logger().warn("Using base model only (LoRA loading failed)")
          self.get_logger().warn("Your fine-tuned weights are NOT loaded!")
          self.model = base_model
          
      except Exception as e:
          self.get_logger().error(f"Critical error loading model: {e}")
          raise e
  
  def rebuild_lora_config(self, config_path):
      """Rebuilding model from started"""
      try:
          # Read the orignall parameter
          with open(config_path + '.backup', 'r') as f:
              original_config = json.load(f)
          
          # Creating a Minimal Model
          minimal_config = {
              "peft_type": "LORA",
              "task_type": "CAUSAL_LM",
              "r": original_config.get("r", 8),
              "lora_alpha": original_config.get("lora_alpha", 16),
              "lora_dropout": original_config.get("lora_dropout", 0.1),
              "target_modules": original_config.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
              "bias": original_config.get("bias", "none"),
              "fan_in_fan_out": original_config.get("fan_in_fan_out", False),
              "init_lora_weights": original_config.get("init_lora_weights", True),
              "base_model_name_or_path": original_config.get("base_model_name_or_path", self.base_model_path)
          }
          
          # Save the rebuild configuration
          with open(config_path, 'w', encoding='utf-8') as f:
              json.dump(minimal_config, f, indent=2)
          
          self.get_logger().info("Config rebuilt with minimal standard parameters")
          return True
          
      except Exception as e:
          self.get_logger().error(f"Error rebuilding config: {e}")
          return False
  
  def generate_response(self, prompt, max_length=512, use_sampling=False):
      """Generate response"""
      if self.model is None:
          return "Model not loaded yet. Please wait..."
      
      model_type = "LoRA Fine-tuned" if hasattr(self.model, 'peft_config') else "Base Model"
      input_text = f"User: {prompt}\n\nAssistant:"
      try:
          inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
          
          generation_config = {
              "max_new_tokens": max_length,
              "eos_token_id": self.tokenizer.eos_token_id,
              "pad_token_id": self.tokenizer.pad_token_id,
              "repetition_penalty": 1.1,
          }
          
          if use_sampling:
              generation_config.update({
                  #"do_sample": False,
                  "do_sample": True,
                  "temperature": 0.7,
                  "top_p": 0.9,
                  "top_k": 50,
              })
          else:
              generation_config["do_sample"] = False
          
          with torch.no_grad():
              outputs = self.model.generate(**inputs, **generation_config)
          
          input_length = inputs['input_ids'].shape[1]
          generated_tokens = outputs[0][input_length:]
          response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                  
          return response.strip()
          
      except Exception as e:
          self.get_logger().error(f"Error generating response: {e}")
          return f"Error generating response: {str(e)}"
  
  def query_callback(self, msg):
      """Deal with the check from ros2 topic"""
      if self.model is None:
          self.get_logger().warn("Model not loaded yet!")
          error_msg = String()
          error_msg.data = "Model not loaded yet. Please wait for model loading to complete."
          self.response_publisher.publish(error_msg)
          return
          
      query = msg.data
      self.get_logger().info(f"Received query: {query}")
      
      try:
          #  Generate response
          response = self.generate_response(query)
          
          #  Publish response
          response_msg = String()
          response_msg.data = response
          self.response_publisher.publish(response_msg)
          
          self.get_logger().info(f"Published response: {response[:100]}...")
          
          # save the history
          self.conversation_history.append({
              "timestamp": datetime.now().isoformat(),
              "user": query,
              "assistant": response,
              "source": "ros2_topic"
          })
          
      except Exception as e:
          self.get_logger().error(f"Error processing query: {e}")
          error_msg = String()
          error_msg.data = f"Error processing query: {str(e)}"
          self.response_publisher.publish(error_msg)
  
  def publish_response(self, response_text):
      """Publish the response message"""
      msg = String()
      msg.data = response_text
      self.response_publisher.publish(msg)
      self.get_logger().info(f"Published: {response_text[:50]}...")
  
  def chat(self):
      """starting chat"""
      # Loading Models in Background Threads for ROS2 (Non-Blocking Approach)
      def load_model_thread():
          try:
              self.load_model()
          except Exception as e:
              self.get_logger().error(f"Failed to load model: {e}")
              print(f"Failed to load model: {e}")
      
      model_thread = threading.Thread(target=load_model_thread, daemon=True)
      model_thread.start()
      
      print("\n" + "="*60)
      print("Trajectory Assistant Chat Interface with ROS2")
      print("="*60)
      print("Commands:")
      print("  'quit' or 'exit' - Exit chat")
      print("  'clear' - Clear conversation history")
      print("  'history' - Show conversation history")
      print("  'save' - Save conversation to file")
      print("  'sampling on/off' - Toggle sampling mode")
      print("  'publish <message>' - Publish message to ROS2 topic")
      print("  'status' - Check model loading status")
      print("  'config' - Show LoRA config info")
      print("  'model-info' - Show detailed model information")
      print("="*60)
      print("ROS2 Topics:")
      print("  Subscribes to: /assistant_query")
      print("  Publishes to: /assistant_response")
      print("="*60)
      print("Model is loading in background...")
      
      sampling_mode = False
      
      while True:
          try:
              user_input = input("\nYou: ").strip()
              
              if not user_input:
                  continue
                  
              # Processing Instructions 
              if user_input.lower() in ['quit', 'exit', 'q']:
                  print("Goodbye!")
                  break
              elif user_input.lower() == 'clear':
                  self.conversation_history.clear()
                  print("Conversation history cleared!")
                  continue
              elif user_input.lower() == 'history':
                  self.show_history()
                  continue
              elif user_input.lower() == 'save':
                  self.save_conversation()
                  continue
              elif user_input.lower() == 'sampling on':
                  sampling_mode = True
                  print("Sampling mode enabled")
                  continue
              elif user_input.lower() == 'sampling off':
                  sampling_mode = False
                  print("Sampling mode disabled (greedy decoding)")
                  continue
              elif user_input.lower().startswith('publish '):
                  message = user_input[8:]  
                  self.publish_response(message)
                  continue
              elif user_input.lower() == 'status':
                  self.show_status()
                  continue
              elif user_input.lower() == 'config':
                  self.show_config_info()
                  continue
              elif user_input.lower() == 'model-info':
                  self.show_model_info()
                  continue
              
              # Checking if the Model is Loaded
              if self.model is None:
                  print("Model is still loading. Please wait or type 'status' to check.")
                  continue
              
              # Generate response
              print("Assistant: ", end="", flush=True)
              response = self.generate_response(user_input, use_sampling=sampling_mode)
              print(response)
              
              # publish to the ROS2 topic
              self.publish_response(response)
              
              # save to the history
              self.conversation_history.append({
                  "timestamp": datetime.now().isoformat(),
                  "user": user_input,
                  "assistant": response,
                  "sampling_mode": sampling_mode,
                  "source": "terminal_chat"
              })
              
          except KeyboardInterrupt:
              print("\nGoodbye!")
              break
          except Exception as e:
              print(f"Error: {e}")
  
  def show_status(self):
      """show the information"""
      if self.model is None:
          print("❌ Model is still loading...")
      else:
          is_lora = hasattr(self.model, 'peft_config')
          if is_lora:
              print("Model loaded successfully!")
              print("Using LoRA fine-tuned model (your custom weights are active)")
          else:
              print("Model loaded successfully!")
              print("Using BASE model only (your fine-tuned weights are NOT loaded)")
          print(f"Model type: {type(self.model).__name__}")
  
  def show_model_info(self):
      """显示模型详细信息"""
      if self.model is None:
          print("Model not loaded yet")
          return
          
      print("\n" + "="*50)
      print("MODEL INFORMATION")
      print("="*50)
      
      is_lora = hasattr(self.model, 'peft_config')
      print(f"Model Type: {type(self.model).__name__}")
      print(f"LoRA Active: {'✅ YES' if is_lora else '❌ NO'}")
      
      if is_lora:
          print("Your fine-tuned weights ARE being used!")
          try:
              peft_config = self.model.peft_config
              for adapter_name, config in peft_config.items():
                  print(f"\nAdapter: {adapter_name}")
                  print(f"  - r: {config.r}")
                  print(f"  - lora_alpha: {config.lora_alpha}")
                  print(f"  - lora_dropout: {config.lora_dropout}")
                  print(f"  - target_modules: {config.target_modules}")
          except Exception as e:
              print(f"Error reading LoRA config: {e}")
      else:
          print("Your fine-tuned weights are NOT being used!")
          print("Running with base llama-8B only")
      
      print("="*50)
  
  def show_config_info(self):
      config_path = os.path.join(self.lora_model_path, 'adapter_config.json')
      backup_path = config_path + '.backup'
      
      print("\n" + "="*50)
      print("CONFIG FILE INFORMATION")
      print("="*50)
      
      if os.path.exists(config_path):
          try:
              with open(config_path, 'r') as f:
                  config = json.load(f)
              print("Current adapter_config.json:")
              for key, value in config.items():
                  print(f"  {key}: {value}")
          except Exception as e:
              print(f"Error reading current config: {e}")
      else:
          print("No adapter_config.json found")
          
      if os.path.exists(backup_path):
          print(f"Backup available: {backup_path}")
          try:
              with open(backup_path, 'r') as f:
                  backup_config = json.load(f)
              print("Original config (backup):")
              for key, value in backup_config.items():
                  print(f"  {key}: {value}")
          except Exception as e:
              print(f"Error reading backup config: {e}")
      
      print("="*50)
  
  def show_history(self):
      if not self.conversation_history:
          print("No conversation history")
          return
      
      print("\nConversation History:")
      print("-" * 50)
      for i, conv in enumerate(self.conversation_history, 1):
          print(f"{i}. [{conv['timestamp']}] Source: {conv.get('source', 'unknown')}")
          print(f"   You: {conv['user']}")
          print(f"   Assistant: {conv['assistant'][:100]}...")
          if 'sampling_mode' in conv:
              print(f"   Mode: {'Sampling' if conv['sampling_mode'] else 'Greedy'}")
          print("-" * 50)
  
  def save_conversation(self):
      if not self.conversation_history:
          print("No conversation to save")
          return
      
      filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
      try:
          with open(filename, 'w', encoding='utf-8') as f:
              json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
          print(f"Conversation saved to {filename}")
      except Exception as e:
          print(f"Error saving conversation: {e}")


def main():
  # Initial the ROS2
  rclpy.init()
  
  # model location, you should change your path
  base_model_path = "/data1/zzt/models/deepseek-r1/"
  lora_model_path = "/data1/zzt/SLM/Llama_8B/Llama_8b_LoRA/"
  
  # Creating an Assistant Node
  assistant = TrajectoryAssistant(base_model_path, lora_model_path)
  
  # Running ROS2 Nodes in Separate Threads
  def ros_spin():
      try:
          rclpy.spin(assistant)
      except Exception as e:
          assistant.get_logger().error(f"ROS2 spin error: {e}")
  
  ros_thread = threading.Thread(target=ros_spin, daemon=True)
  ros_thread.start()
  
  try:
      # start chat
      assistant.chat()
  finally:
      # clean resource
      assistant.destroy_node()
      rclpy.shutdown()


if __name__ == "__main__":
  main()