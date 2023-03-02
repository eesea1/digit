import telebot
import predictor

from datetime import datetime


bot = telebot.TeleBot("6206049118:AAEnZ6rinWSdiVKRHc8wMWk7RgGhkdgML-4")


@bot.message_handler(content_types=["photo", "document"])
def start(message):
    try:
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        name: str = datetime.now().strftime(r"%d/%m/%Y %H:%M:%S").replace(" ", "_").replace("/", "").replace(":", "")
        with open(f"downloads/{name}.png", 'wb') as new_file:
            new_file.write(downloaded_file)
    except:
        print('NOT A DOCUMENT!')
    
    try:
        file_info = bot.get_file(message.photo[0].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        name: str = datetime.now().strftime(r"%d/%m/%Y %H:%M:%S").replace(" ", "_").replace("/", "").replace(":", "")
        with open(f"downloads/{name}.png", 'wb') as new_file:
            new_file.write(downloaded_file)
    except:
        print('NOT A PHOTO!')
    new_file.close()
    
    
    print(name)
    prediction, rz_image = predictor.predict_img(f"downloads\{name}.png") 
    bot.send_message(message.from_user.id, prediction)
    bot.send_photo(message.from_user.id, rz_image)
    
    
def main():
    bot.polling(none_stop=True, interval=0)


if __name__ == "__main__":
    main()