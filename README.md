# Real-Time Context-Aware Conversational Agent 

![UP Logo](https://i.imgur.com/dexJzJr.png)

This project was developed within the framework of the "Natural Language Processing" course offered within the Faculty of Electrical and Computer Engineering, master's program "Computer and Software Engineering", University of Pristina "Hasan Prishtina"

Mentor: Prof. Dr. Mërgim Hoti

Students: Albesë Matoshi, Avdyl Krasniqi, Elon Demi
### About
This project aims to develop a chatbot that answers users' questions about movies in real time.
##### Main Objectives:
- Selection of Dataset and Tokenization
- Contextual Pairing
- Fine Tunning
- Model Evaluation

  ![Steps](https://i.imgur.com/RjZYTrL.png)

### Things good to know about before we start:
The project is heavily based on the libraries provided by Hugging Face, such as
- [Transformers](https://huggingface.co/docs/transformers/index), 
- [Datasets](https://huggingface.co/docs/datasets/index), 
- [Tokenizers](https://huggingface.co/docs/tokenizers/index).
- [TRL (Transformer Reinforcement Learning)](https://huggingface.co/docs/trl/index)

The purpose of these libraries, as well as the understanding of their documentation, methods, and functionalities, enables the reader of this project to better comprehend our code in detail. It also allows them to adjust various parameters according to their specific goal of developing/modifying/training this model further, depending on the computational resources they possess.
The training on our end was conducted on the following machine:

- Dell Inc. Precision 7760
- 11th Gen Intel® Core™ i7-11850H × 16
- 32.0 GiB RAM
- 1.0 TB SSD
- NVIDIA T1200 Laptop GPU / NVIDIA-SMI 535.183.01 / Driver Version: 535.183.01 / CUDA Version: 12.2
- Linux 6.8.0-51-generic Kernel Version



### Selection of dataset
The provided dataset is a collection from the OPUS-OpenSubtitles project, published in 2016 & 2018. This dataset is usable for various research and practical purposes, including machine translation, natural language processing (NLP), or language learning.
The dataset was found in: https://opus.nlpl.eu/OpenSubtitles/en&sq/v2016/OpenSubtitles
#### Dataset Details
The source of this data is the subtitles of movies and series. 

| Number of attributes | Nr. of sentences | Dataset Size                          | 
|----------------------|------------------|---------------------------------------|
| 1 - Sentences        | 1,903,990   | 1,903,990 Sentences 11,328,628 tokens | 





### Tokenization

For easier analysis and processing, a text-processing tool and NLP model was used(Bert Tokenizer). This model  converted raw text into tokens and also handled the words that are not in the vocabulary by breaking them into subwords or characters. After executing the tokenization function the output was saved in a json file for further use.
Before choosing BERT, we compared it to GPT2 and concluded that Bert is more accurate.


##### Code for tokenization:
  ![Bounding box image](https://i.imgur.com/gJLlnJ1.png)

##### Results after data tokenization with BERT:

  ![Bounding box image](https://i.imgur.com/Pe8FF5P.png)

##### Results after data tokenization with GPT2:
  ![Tokenization with GPT2](https://i.imgur.com/3FJR7K7.png)

In the image above we can see that BERT gives more accurate results than GPT2

### Contextual Pairing

To pair each of our input sentence to a meaningful response a contextual pairing method was used. The dataset that we used(OpenSubtitles), contains sentences that are sequential, that is the reason why extraction of meaningful dialogue pairs was necessary to structure the data properly before feeding it into the model. After executing the contextual pairing function the output was saved in a json file for further use.

##### Code for contextual pairing:
  ![Bounding box image](https://i.imgur.com/zXO9s56.png)

##### Results after contextual pairing:

  ![Bounding box image](https://i.imgur.com/OJI6yqq.png)


### Fine Tuning

Before performing fine-tuning, it is necessary to select one of the pre-trained models. Generally, there are several different [pre-trained models](https://www.kaggle.com/models) provided by OpenAI, Meta, Google, etc.

In our case, we have chosen model [3.2B](https://www.kaggle.com/models/metaresearch/llama-3.2) offered by Meta for the following reasons:
- It has an open license for study purposes [(Llama 3.2 Community License)]((https://www.kaggle.com/models/metaresearch/llama-3.2)).
- It offers various versions, adaptable for devices with both low and high processing power. 
- The LLaMa model is designed for text, unlike other models such as Segment Anything, which is dedicated to images.

For better training our model to recognise response patterns and give accurate responses for movie subtitles we have used a pre-training model(fine tuning) called Llama 3.  Supervised Fine-Tuning Task was employed to train the model on specific labeled dataset that we created by creating contextual pairs.




### Model Evaluation
As the last step, to check if the dataset is propery fine-tuned, model evaluation was done by using the library wandb, which enables tracking of metrics.
We got the following results from [wandb](https://wandb.ai):

  
| Parameter                 | Value              | Description                                                                                                                                                                                    |
|---------------------------|--------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Eval / Loss               | 2.66126            | This represents the evaluation loss, which measures the error of the model on the validation dataset. A lower value indicates better performance.                                              |
| Eval / Runtime            | 76.3163            | The total time (in seconds) taken to evaluate the model on the validation dataset.                                                                                                             |
| Eval / samples_per_second | 1.66               | The number of samples evaluated per second during the evaluation phase.                                                                                                                        |
| Eval / steps_per_seconds  | 1.16               | The number of evaluation steps processed per second.                                                                                                                                           |
| Total_flos                | 9167114454148552.0 | Total floating-point operations (FLOPs) performed by the model during training and evaluation. This is a measure of computational effort.                                                      |
| Train / Epoch             | 1.0                | The current epoch number during training. An epoch represents one complete pass through the entire training dataset.                                                                           |
| Train / global_step       | 450                | The total number of optimization steps completed. This includes all the steps across all epochs.                                                                                               |
| Train / grad_norm         | 2.54422            | The norm of the gradients, which measures the magnitude of changes being applied to the model's weights during backpropagation. Large values may indicate instability or issues with learning. |
| Train / learning_rate     | 0.0                | The current learning rate being used by the optimizer during training. A value of 0.0 may indicate the end of training or a specific learning rate schedule.                                   
                                                                                                                                                   |
| Train / loss              | 2.7479             | The loss computed during the last training step. It measures how well the model is performing; lower values indicate better performance.                                                       |
| Train_loss                | 2.5901             | The average loss across all training steps up to this point in the training process.                                                                                                                                                                                               |
| Train_runtime             | 1798.8689          | The total time (in seconds) taken to complete the training process.  |
| train_samples_per_second  | 0.5                | The number of training samples processed per second during training. |
| train_steps_per_second    | 0.25               | The number of training steps processed per second during training.|


  ![Model evaluation](https://i.imgur.com/Aoof2Ou.png)

As we can see, during step iterations, the eval/loss and train/grad_norm was dropping, which is a good thing.



### Technology Used

To implement the chatbot, which requires various processes such as data collection, preprocessing, format analysis, training and evaluation, various technologies have been used, such as:
- Transformers – Python Library, which provided many transformed-based models like Bert, GPT.(https://huggingface.co/docs/transformers/en/index)
- BertTokenizer - text processing tool, NLP model developed by google. Was used to convert text into tokens(https://huggingface.co/docs/transformers/en/model_doc/bert) 
- TRL - Transformer Reinforcement Learning, library which provided a set of tools to train models with Supervised Fine-tuning trainer(SFTT)(https://huggingface.co/docs/trl/en/index)
- Weights & Biases (W&B) – tool used for model evaluation.(https://wandb.ai/site/)





## Project Setup

Ensure Python 3.12+ and a virtual environment (venv) are installed before proceeding.

Generate Tokenized Data:
```sh
python3 tokenization.py 
```
Create Contextual Pairing:
```sh
python3 contextual-pairing.py 
```


### Prerequisites about Llama

Although LlaMa 2 is an open-source model, there is still a procedure for which a form must be filled out and the user must agree to the license of this model. To apply for access to this model, 2 forms must be filled out:
https://huggingface.co/meta-llama/Llama-3.2-1B
and https://www.llama.com/llama-downloads/

**Approval often takes up to 2 days from Meta and HuggingFace**


### How to Download the Model

Visit the [Llama repository](https://github.com/) in GitHub where instructions can be found in the Llama README.


### 1. Install the Llama CLI

In your preferred environment, run the command below:

```bash
pip install llama-stack
```
Use -U option to update llama-stack if a previous version is already installed:

```bash
pip install llama-stack -U
```

### 2. Find models list

See latest available models by running the following command and determine the model ID you wish to download:

```bash
llama model list
```
If you want older versions of models, run the command below to show all the available Llama models:
```bash
llama model list --show-all
```


### 3. Select a model

Select a desired model by running:
```bash
llama model download --source meta --model-id MODEL_ID
```

### 4. Specify custom URL

When the script asks for your unique custom URL, please paste the URL below:
```bash
<link provided by META>
```








After completing these steps, you will be able to download the model you applied for and will have the following files in your possession:

```text
total 2422724
-rwx------ 1 avdyl gr2        877 Sep 24 16:09 config.json
-rwx------ 1 avdyl gr2        189 Sep 24 16:09 generation_config.json
-rwx------ 1 avdyl gr2       1519 Sep 24 16:09 .gitattributes
-rwx------ 1 avdyl gr2       7712 Sep 24 16:09 LICENSE.txt
-rwx------ 1 avdyl gr2 2471645608 Sep 24 16:09 model.safetensors
-rwx------ 1 avdyl gr2      34226 Sep 24 16:09 README.md
-rwx------ 1 avdyl gr2         68 Sep 24 16:17 special_tokens_map.json
-rwx------ 1 avdyl gr2      50870 Sep 24 16:17 tokenizer_config.json
-rwx------ 1 avdyl gr2    9084449 Sep 24 16:17 tokenizer.json
-rwx------ 1 avdyl gr2       6021 Sep 24 16:09 USE_POLICY.md
```

The folder containing these files will be used as input during the `llamatune.py` program.

# Interacting with our FIEK chatbot

After you finished setuping the project, tuning the model, you can now interact with trained model using the following command:
```sh
python3 interact.py
```

We also implemented chaining prompt, so it doesn't look like session-less, instead, using the bot itself, when you ask several questions, it will combine the context, and give a new prompt for itself.

Here is an example:

```
User: What happened in season 4 of Mr. Robot?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Chatbot: Season 4 of Mr. Robot premiered on October 7, 2017. Here's a brief summary of the key events:

* The season begins with Elliot Alderson (Rami Malek) and Darlene (Jena Malone) on the run from the FBI and Mr. Robot's (Christian Slater) plans to expose the corporation's true intentions.
* Elliot and Darlene are forced to go on the run, and Elliot becomes increasingly obsessed with finding a way to take down the corporation from the inside.
* Meanwhile, Angela Moss (Carly Chaikin) becomes more involved in Elliot's plans, and their relationship deepens.
* Elliot and Mr. Robot's plan to take down the corporation is revealed, and Elliot becomes more and more conflicted about the morality of their actions.
* The season culminates in a dramatic and intense confrontation between Elliot and the corporation, with Elliot ultimately deciding to go rogue and take matters into his own hands.

Overall, Season 4 of Mr. Robot is a wild and unpredictable ride, with many twists and turns that keep viewers on the edge of their seats.

```

Now, when the user asks the second question:

```
User: What is Dissociative identity disorder?
```

We programmed the chatbot as the following prompt:


Someone first asked ***QUESTION A***, and then followed up with question ***QUESTION B***. answer the second question and relate it to the first question.


```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You show only the direct answer, do not say here is what you have asked for or anything like that.<|eot_id|><|start_header_id|>user<|end_header_id|>

Someone first asked "What happened in season 4 of Mr. Robot?", and then followed up with question "What is Dissociative identity disorder?". answer the second question and relate it to the first question.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

Then the chatbot gives us a contextual answer:

```
Dissociative identity disorder (DID) is a mental health condition characterized by the presence of two or more distinct identities or personalities within an individual, often with different thoughts, feelings, and behaviors. People with DID may experience a range of symptoms, including:

* Memory lapses or gaps
* Difficulty forming and maintaining relationships
* Emotional instability and mood swings
* Dissociation from one's surroundings or self
* Intrusive thoughts or flashbacks
* Identity fragmentation

In the context of Mr. Robot, the character Elliot Alderson's dissociative identity disorder is a central plot point. As Elliot navigates the complex web of corporate espionage and conspiracy, he experiences dissociative episodes, where he switches between different personalities or identities, often with conflicting goals and motivations. This dissociation allows Elliot to cope with the trauma and stress of his situation, but it also creates challenges for him to maintain his sense of self and identity.

The show's creator, Sam Esmail, has stated that he drew inspiration from his own experiences with dissociative identity disorder when developing the character of Elliot Alderson. The show explores the psychological and emotional toll of DID, as well as the complexities of identity and selfhood, which are also central themes in the show.

```
