import utils
import csv
import time

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
}


def main():
    urls = []

    with open('raw_data/furniture stores pages.csv') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            urls.append(row[0])

    sites_text, sites_status = utils.urls_to_texts(urls, headers)

    print("Status: text and status extraction ended!")

    sites_products = []

    for i, text in enumerate(sites_text):
        product_list = utils.model_predict(text, 900)

        sites_products.append(product_list)

        print(f"Status: {i} prediction ended!")

    utils.save_outputs(urls, sites_status, sites_products)


if __name__ == '__main__':
    main()