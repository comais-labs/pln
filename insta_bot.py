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
postsposts_elements = driver.find_elements(By.CLASS_NAME, '_aagw')
#posts_elements = driver.find_elements(By.XPATH, '//a[contains(@href, "/p/")]')

while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
    time.sleep(5)
    #new_posts = driver.find_elements(By.XPATH, '//a[contains(@href, "/p/")]')
    new_posts = driver.find_elements(By.CLASS_NAME, '_aagw')

    if len(new_posts) == len(posts_elements):
        break
    posts_elements = new_posts
    
posts = posts_elements
for post in posts:
    post.click()
    time.sleep(2)
    ele_comentarios = driver.find_elements(By.CLASS_NAME, '_a9zo')
    for comentario in ele_comentarios[1:]:
        autor = comentario.find_element(By.TAG_NAME, 'h3').text
        texto = comentario.find_element(By.CLASS_NAME, '_a9zs').text
        print('Autor:', autor)
        print('Comentario:', texto)
    close = driver.find_element(By.CLASS_NAME,'x160vmok')
    close.click()
    time.sleep(2)

    
