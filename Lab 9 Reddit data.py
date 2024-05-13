import requests

def retrieve_top_posts():
    url = "https://hacker-news.firebaseio.com/v0/topstories.json"
    response = requests.get(url)

    if response.status_code == 200:
        top_story_ids = response.json()[:10]  # Get the top 100 story IDs
        top_posts = []
        for story_id in top_story_ids:
            story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
            story_response = requests.get(story_url)
            if story_response.status_code == 200:
                top_posts.append(story_response.json())
            else:
                print(f"Failed to retrieve story with ID {story_id}. Status code:", story_response.status_code)
        return top_posts
    else:
        print("Failed to retrieve top stories. Status code:", response.status_code)
        return None

top_posts = retrieve_top_posts()
if top_posts:
    for post in top_posts:
        print("Title:", post['title'])
        print("URL:", post.get('url', 'N/A'))
        print("Score:", post['score'])
        print("-----")