import pandas as pd
import numpy as np


def main():
    WORDS_PER_FILE = 300

    df = pd.read_parquet('raw_data/furniture_stores_parquet.parquet')
    site_contents = np.array(df['text'])

    file_counter = 0
    file = None

    for text in site_contents[1:]:
        if file_counter >= 250:
            break
        if text != "Unreachable Url":

            word_split = text.split(' ')

            for i, word in enumerate(word_split):
                if i % WORDS_PER_FILE == 0:
                    if file:
                        file.close()

                    if file_counter >= 250:
                        break

                    file = open(f'raw_data/batched_text/text_set_{file_counter}.txt', 'w')
                    file_counter += 1

                file.write(f'{word} ')

            file.close()


if __name__ == '__main__':
    main()
