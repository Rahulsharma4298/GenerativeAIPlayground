from typing import List

import requests
from langchain_core.tools import tool
from mypy_extensions import TypedDict


class Med(TypedDict):
    name: str
    label: str
    price: str
    rx_required: str
    image: str
    url: str
    ratings: str

@tool
def search_medicine(query: str, city: str="Indore", max_items: int=5) -> List[Med]:
    """ Use it to search for medicine or health related products.
    You can also suggest medicine for user problem.
    If user talks about buying a health supplements, use this tool to search it.
    User can search items such as medicine, condoms and supplements.
    Show all the information you get in context as Markdown.
    Show url as Buy now button links to 'url'.
    Show image as Markdown.
    Args:
        query: The search query for medicine
        city: Optional, default is 'Indore'
        max_items: Optional, No. of items as response, default is 5
    """
    headers = {
        'accept': 'application/vnd.healthkartplus.v4+json',
        'accept-language': 'en-US,en;q=0.9,hi;q=0.8',
        'x-city': f'{city}',
    }
    response = requests.get(
        f'https://www.1mg.com/pwa-api/api/v4/search/all?q={query}&city={city}&filter=&page_number=0&scroll_id=&per_page=10&types=sku,allopathy&sort=relevance&fetch_eta=true&is_city_serviceable=true',
        # cookies=cookies,
        headers=headers,
    )
    if response.ok:
        formatted_response = []
        data = response.json()['data']
        for item in data['search_results']:
            if item.get('prices') is None:
                continue
            result = {'name': item['name'],
                      'label': item['label'],
                      'price': item.get('prices', {}).get('discounted_price'),
                      'rx_required': item.get('rx_required'),
                      'image': item.get('image'),
                      'url': "http://1mg.com/"+item.get('url') ,
                      'ratings': item.get('ratings', {})['average_rating'] if item.get('ratings', {}) is not None else None
                      }
            print(result)
            formatted_response.append(result)
        return formatted_response[:max_items]
    response.raise_for_status()

if __name__ == '__main__':
    search_medicine('fluoxetine')