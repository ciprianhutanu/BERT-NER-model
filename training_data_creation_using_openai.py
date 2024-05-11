import time
import openai

openai.api_key = '<KEY>'

BATCHES = 250

role_introduction = {'role': 'system',
             'content': 'You are my assistant "Chip". You will receive a series of texts extracted from some websites \
             that host furniture store sales. Your role is to indentify the products in the text. Keep in mind, they \
             will be furniture products. You will write only the products on separate lines. If you dont find any, \
             its ok, you will respond with the text "None". I expect only those formats, and nothing more. I dont \
             want any product category. Dont use "-" to list products, i dont need any formating made by you. \
             Countries are not furniture.'}


def creating_all_the_data_from_batches():
    messages = [role_introduction]

    file_counter = 43

    for i in range(89, BATCHES):
        read_file = open(f'raw_data/batched_text/text_set_{i}.txt', 'r')
        text = read_file.read()
        read_file.close()

        messages.append({'role': 'user', 'content': text})

        chat = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=messages)

        answer = chat.choices[0].message.content

        messages.append({'role': 'assistant', 'content': answer})

        if answer != 'None':
            file_writer(f'./training_data/each_data_set/data_set_{file_counter}.txt', text, answer)
            file_counter += 1

            print(answer.split('\n'))

        print("Progress made so far: " + str((i+1)*100/BATCHES) + "%")
        if len(messages) > 15:
            # clearing the buffer, letting the first 7 messages as they will be the accurate ones
            messages = messages[0:7]

        time.sleep(2)



def converting_only_one_batch(batch_number: int):
    messages = [role_introduction]

    read_file = open(f'../raw_data/batched_text/text_set_{batch_number}.txt', 'r')
    text = read_file.read()

    messages.append({'role': 'user', 'content': text})

    read_file.close()

    chat = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=messages)

    answer = chat.choices[0].message.content

    write_file = open(f'../training_data/each_data_set/dataset_{batch_number}.txt', 'w')
    write_file.write(answer)
    write_file.close()


def file_writer(file_address: str, text: str, answer: str):
    file = open(file_address, 'w')

    products = answer.split('\n')
    products = [[word.upper() for word in prod.split(' ') if word != '-'] for prod in products]

    beginnings = set([prod[0] for prod in products])
    in_product = []
    for prod in products:
        if len(prod) > 1:
            in_product.extend(prod[1:])

    in_product = set(in_product)

    split_text = text.split(" ")

    for word in split_text:
        if word != "":
            if word.upper() in beginnings:
                file.write(f'{word} B-PRODUCT\n')
            elif word.upper() in in_product:
                file.write(f'{word} I-PRODUCT\n')
            else:
                file.write(f'{word} O\n')

    file.close()


if __name__ == '__main__':
    creating_all_the_data_from_batches()