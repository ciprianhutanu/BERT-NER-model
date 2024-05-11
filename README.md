# BERT-NER model

The project consist of a **BERT model** trained to identify products displayed on furniture stores websites. 


## Creating the dataset with *OpenAI*

To start, I received a *.csv* file containing about 700 websites that are supposed to sell furniture articles. The sites are not guaranteed to work, some not existing anymore, some not responding. I extracted all the text from the websites using `BeautifulSoup` library and when the domain wasn't existing I labeled them as *"Unreachable Url"*.

With all of the data, I had created 250 batches of text of maximum 300 words. Then, using `openai` library, I trained `GPT-3.5 Turbo` model to identify the products in that batches. As the instruction message will suggest, working with it wasn't really a walk in the park. :)
> You are my assistant "Chip". You will receive a series of texts extracted from some websites that host furniture store sales. Your role is to identify the products in the text. Keep in mind, they will be furniture products. You will write only the products on separate lines. If you dont find any, its ok, you will respond with the text "None". I expect only those formats, and nothing more. I dont want any product category. Dont use "-" to list products, i dont need any formating made by you. Countries are not furniture.'}

It kept on changing the text format, or on considering, for some reason, countries as furniture. After a while, I gave up with the idea of only having products, and i let him respond with categories also. After all, if a site contains the *outside tables* category, it is understood that they also sell outside tables, so it can be recognized as products.

Follow the links for:

 - [Website csv](https://github.com/ciprianhutanu/BERT-NER-model/blob/main/input.csv)
 - [Websites text extraction](https://github.com/ciprianhutanu/BERT-NER-model/blob/main/web_scraping.ipynb)
 - [Batch creation](https://github.com/ciprianhutanu/BERT-NER-model/blob/main/data_modeling.py)
 - [Text batches](https://github.com/ciprianhutanu/BERT-NER-model/tree/main/raw_data/batched_text)
 - [Product extraction with openai](https://github.com/ciprianhutanu/BERT-NER-model/blob/main/training_data_creation_using_openai.py)

## Labeling the text

When I had the products, it remained to label them. For that I used the following logic:

 - The first word: `B-PRODUCT`
 - The rest of the product: `I-PRODUCT`
 - Any other word: `O`
 
 So, for a sentence like: *"Ciprian nightstand 2 drawers is priced at 99.99"*, I would end up with something like this:
 
| Word | Label |
|--|--|
| Ciprian | B-PRODUCT |
| nightstand  | I-PRODUCT |
| 2 | I-PRODUCT |
| drawers  | I-PRODUCT |
| is | O |
| priced  | O |
| at  | O |
| 99.99 | O |

Considering that, with all the products I got for a batch, I created two sets: `beginnings` and `in_product`. When I got a word that was in beginnings, I labeled it `B-PRODUCT`, in in_product `I-PRODUCT`, else `O`.

If a batch didn't include any product, I chose not to include them in the datasets.

Follow the links for:

- [The datasets](https://github.com/ciprianhutanu/BERT-NER-model/tree/main/training_data/each_data_set)
- [Labeling function](https://github.com/ciprianhutanu/BERT-NER-model/blob/main/training_data_creation_using_openai.py)( `file_writer(file_address, text, answer)` where **file_address** is the address of the respective dataset, **text** is the batch text and **answer** is the response from gpt)
 

## Training

With the datasets ready, it remained to train the model. For that I used the `transformers` library, with `PyTorch` implementation. I split the datasets into 100 training sets and 26 testing sets. Being the newest and the recommended one, I used the `distilbert` model. I chose the epoch evaluation strategy, and set them to 100. Also in an attempt to avoid over-fitting, I set the weight decay to 10 to the power of -6.

The results are great when comes down to recognizing the products names. When tasked with the previous example (*"Ciprian nightstand 2 drawers is priced at 99.99"*), the model returned the following result:

![Result](https://github.com/ciprianhutanu/BERT-NER-model/blob/main/example_result.png)

Follow the links for:

 - [Model training](https://github.com/ciprianhutanu/BERT-NER-model/blob/main/model_training.ipynb)
 - [Results](https://github.com/ciprianhutanu/BERT-NER-model/blob/main/output.parquet)
