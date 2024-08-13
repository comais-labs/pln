import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import os
from dotenv import load_dotenv

load_dotenv()
PASSWORD = os.environ.get('PASSWORD')

driver = webdriver.Chrome()
driver.implicitly_wait(10)
driver.get('https://www.instagram.com/')
time.sleep(5)

username = driver.find_element(By.NAME, 'username')
password = driver.find_element(By.NAME, 'password')
username.send_keys('roger_nogueira')
password.send_keys(PASSWORD)
password.send_keys(Keys.RETURN)
time.sleep(5)
driver.get('https://www.instagram.com/ppggtd_uft/')
time.sleep(2)
#postsposts_elements = driver.find_elements(By.CLASS_NAME, '_aagw')
posts_elements = driver.find_elements(By.XPATH, '//a[contains(@href, "/p/") or contains(@href, "/reel/")]')

posts_urls = []
while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
    time.sleep(10)
    new_posts = driver.find_elements(By.XPATH, '//a[contains(@href, "/p/") or contains(@href, "/reel/")]')
    #new_posts = driver.find_elements(By.CLASS_NAME, '_aagw')

    if len(new_posts) == len(posts_elements):
        break
    if len(posts_elements) >= 10:
        break
    posts_elements = new_posts
 
posts_urls = [post.get_attribute('href') for post in posts_elements]    
posts = posts_elements
dict_posts = []
for post in posts:
    post.click()
    #driver.get(post)
    time.sleep(3)
    ele_comentarios = driver.find_elements(By.CLASS_NAME, '_a9zo')
    if len(ele_comentarios) > 1:
        comentarios = []    
        for comentario in ele_comentarios[1:]:
            autor = comentario.find_element(By.TAG_NAME, 'h3').text
            texto = comentario.find_element(By.CLASS_NAME, '_a9zs').text
            print('Autor:', autor)
            print('Comentario:', texto)
            comentarios.append({'autor': autor, 'comentario': texto})
    else:
        comentarios = []
    dict_posts.append({
        'url':post.get_attribute('href'), 
        'comentarios': comentarios
    })
    close = driver.find_element(By.CLASS_NAME,'x160vmok')
    close.click()
    time.sleep(2)


row=[]
for post in dict_posts:
    for comentario in post['comentarios']:
        row.append({'url':post['url'], 'autor':comentario['autor'], 'comentario':comentario['comentario']})

df = pd.DataFrame(row)
df.to_csv('comentarios.csv', index=False)

    
