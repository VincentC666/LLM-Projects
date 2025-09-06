import json
import requests
from bs4 import BeautifulSoup
import os

def fetch_and_parse(url):
    # Request the url
    response = requests.get(url)
    response.encoding = 'utf-8'

    # parse the url
    soup = BeautifulSoup(response.text, 'html.parser')

    #extract all div with class = 'section'
    section_divs = soup.find_all('div', class_='section')

    data = {}
    for div in section_divs:
        if div:
            # Combine the section number and section name
            header = div.find('h4').get_text(strip=True)
            tag = div.find('a')
            sec_name = tag.get('name') if tag else ''
            header = sec_name + ' ' + header


            paragraphs = div.find_all('p')
            content = ''
            for p in paragraphs:
                # The section number is already exists in the header
                bold_tag = p.find('b')
                if bold_tag:
                    bold_tag.extract()

                # Extract all contents under the section
                text_parts = []
                for text in p.get_text().split():
                    text_parts.append(text)
                content += " ".join(text_parts).strip() + " "
            content = content.strip()
            data[header] = content

    json_output = json.dumps(data, indent=4)
    return json_output

if __name__ == '__main__':
    url = 'https://www.bclaws.gov.bc.ca/civix/document/id/complete/statreg/00_96113_01'
    json_output = fetch_and_parse(url)

    output_Path = 'data'
    file_Name = ('Employment_Standard''.json')
    if not os.path.exists(output_Path):
        os.makedirs(output_Path)

    file_Path = os.path.join(output_Path, file_Name)
    with open(file_Path,'w') as json_file:
        json_file.write(json_output)
    print(f"JSON file {file_Name} saved to {file_Path}")


