---
title: Review of NLP tasks using Hugging Face
tags: [python, tutorial, llm]
image: /assets/images/nlp_tasks_review/llm_tree.png
date: 2025-05-23
pinned: true
type: blog
---
>>The full guide of huggingface is found here [[1]](https://huggingface.co/learn/llm-course/chapter1/1?fw=pt). This is a review of the main concepts along with some codes.

# Introduction

* Natural language processing (NLP) refers to the field focused on enabling computers to understand, interpret and generate human language.
* Large language models (LLMs) is a collective term of NLP models that are trained on a massive dataset. Models example are Llama and Generative Pretrained Transformers (GPTs).

Some NLP tasks include:
1. Classifying the whole senteces - e.g., sentiment analysis
2. Classifying individual word - e.g., named entity extraction
3. Question and answering - given a question and context, extract a factually correct answer.
4. Generating text content
5. Translation

Development of LLMs also means that we can now have seemingly all-knowing chatbot. However, some of the problems associated with LLMs are:
* Hallucination - where the model will provide with ostensibly correct-sounding answer, that is in fact wrong.
* Bias
* Lack of reasoning
* Computational resources

# Transformers

Transformers is a type of architecture that underlies many of the well-known NLP models. This video by [3Blue1Brown](https://www.youtube.com/watch?v=wjZofJX0v4M) explains it quite well.

In a nutshell, a transformer is a deep neural network consisted of several layers, with each layer feeding information to the next. In the case of the GPT, a piece of text is fed to the transformer, and the transformer will output the next most probable word based on the input of all previous words.

The `transformer` library of huggingface is very versatile. For example, the `pipeline` object allows one to use a variety of pretrained models for different tasks, including:

* `text-generation`
* `text-classification`
* `summarization`
* `translation`
* `zero-shot-classification`
* `feature-extraction`
* `image-to-text`
* `image-classfication`
* `object-detection`
* `automatic-speech-recognition`
* `audio-classification`
* `text-to-speech`
* `image-text-to-text`

The general code syntax is

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.")
```
**Output**
```
[{'label': 'POSITIVE', 'score': 0.9925463795661926}]
```

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
classifier(
    "Four legged animal with fur, and can be found mummified in egypt",
    candidate_labels=["dog","cat","chicken"],
)
```
**Output**
```
{'sequence': 'Four legged animal with fur, and can be found mummified in egypt',
 'labels': ['dog', 'cat', 'chicken'],
 'scores': [0.3621610403060913, 0.35874396562576294, 0.27909499406814575]}
 ```

*Sometimes it doesn't work as you expect*
## Transformers architecture

Transformers are *language models*, meaning they can be trained in a self-supervised manner. In a nutshell, self-supervised learning is a type of learning where the objective is computed from the input. For example, you crop a part of the image and you let the model predict that part of the image. In the case of NLP, this can be either predicting the next word after reading *n* numbers of words (*causal language modeling*), or predicting a masked input (*masked language modeling*). These are also known as *auxiliary tasks* or pretext tasks; you don't really care about the performance of these tasks, but rather you use this as a pretext for the model to learn about intrisic relationship between the input data (e.g., semantic relationship between words or positions of the pixels within the image) [[3]](https://lilianweng.github.io/posts/2019-11-10-self-supervised/). These pretrained models are useful when are subsequently fine-tuned (using transfer learning) to a specific tasks.

> Most of the transformers use the following three architecture: encoder-only, decoder-only or encoder-decoder (sequence to sequence).

![](/assets/images/nlp_tasks_review/llm_tree.png)
*Image from [Yang et al. 2023](https://arxiv.org/pdf/2304.13712)*

In a nutshell, the difference between the three are as follows:
* Encoder only -> This type of model is useful when the task requires understanding the whole input. These models are characterised as having "bidirectional" attention, because the numerical representation of a word is "influenced" by the adjacent words on both side. This can be achieved by using a *masked language modeling*. These models are known as *auto-encoding*  models. Encoder models are best suited for sentence classification, such as named entity recognition, and extractive question answering. Examples include BERT and DistilBERT.
<p align="centre">
    <img src="https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb8fc5971-d59b-4d5a-bb2a-d8ac6d2aa23a_2022x1224.png" alt="masked language modeling" style="width: 50%; height: auto;" />
</p>

*Example of masked language modeling. Image taken from [Raschka, 2023](https://magazine.sebastianraschka.com/p/understanding-encoder-and-decoder)*

* Decoder only -> This type of model is useful when the task requires predicting the next word. These models are "unidirectional", i.e., the only context the attention layer receives for a word comes from the words preceding it. This model is known as *auto-regressive*.
    >>Modern LLMs mostly use decoder only architecture. Firstly, they are trained to predict the next token from the vast amount of data. Secondly, they are fine-tuned to a specific tasks. For example, text summarisation, question answering, code generation and so on.

* Encoder-Decoder -> This is the combination of the two. The input to the encoders are "converted" to numerical vector representation, which are then used as inputs to the decoder. The decoder in turn will use the information provided by the encoder and will generate the next token in the sentence.
<p align="centre">
    <img src="https://arxiv.org/html/1706.03762v7/extracted/1706.03762v7/Figures/ModalNet-21.png" alt="Vaswani et al., 2017" style="width: 50%; height: auto;" />
</p>

*Illustration of taken from [Vaswani et al., 2017](https://magazine.sebastianraschka.com/p/understanding-encoder-and-decoder)*

Best way to visualise this is in the translation tasks. If the sentence "Hi, my name is Hai" is fed to the encoder, the numerical representation will be added to the decoder along with a start sentence token. The decoder will then use the token and generate "Xin chào, tôi tên là Hải", word by word until it receives a stop token. Example models are BART and T5.

The following table summarises the suggested architecture for each task. Taken from [[4]](https://huggingface.co/learn/llm-course/en/chapter1/6?fw=pt):

Task | Suggested Architecture | Examples
---|---|---
Text classification | Encoder | BERT, RoBERTA
Text generation | Decoder | GPT, Llama 
Translation | Encoder-Decoder | T5, BART 
Summarisation | Encoder-Decoder | BART, T5 
Named entity recognition | Encoder | BERT, RoBERTA
QA (extractive)  | Encoder | BERT, RoBERTA
QA (generative) | Encoder-Decoder or Decoder| T5, GPT
Conversational AI | Decoder | GPT, Llama

## How does inference work?

Inference is the way in which the 

