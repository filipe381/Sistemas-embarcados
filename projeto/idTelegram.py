from telegram import Bot
import asyncio

# Substitua por seu token válido do Telegram Bot
BOT_TOKEN = '7390853298:AAFEZhfjtMaB7lfNGk4NjjWsQxNYp42olUs'

async def get_chat_id():
    return 5721801316

# Teste a função assincronamente
if __name__ == "__main__":
    chat_id = asyncio.run(get_chat_id())
    print(f"Chat ID: {chat_id}")
