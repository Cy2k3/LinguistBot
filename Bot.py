from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from langdetect import detect, DetectorFactory
from transformers import MarianMTModel, MarianTokenizer
import json
import logging

# Ensure consistent language detection
DetectorFactory.seed = 0

# Supported languages
supported_languages = ['en', 'uk', 'pl']
language_names = {'en': "English", 'uk': "Ukrainian", 'pl': "Polish"}

# Translation models
translation_models = {
    ('en', 'uk'): "Helsinki-NLP/opus-mt-en-uk",
    ('uk', 'en'): "Helsinki-NLP/opus-mt-uk-en",
    ('en', 'pl'): "Helsinki-NLP/opus-mt-en-pl",
    ('pl', 'en'): "Helsinki-NLP/opus-mt-pl-en",
    ('uk', 'pl'): "Helsinki-NLP/opus-mt-uk-pl",
    ('pl', 'uk'): "Helsinki-NLP/opus-mt-pl-uk"
}

# Cache for models and tokenizers
model_cache = {}
tokenizer_cache = {}

# Store user language preferences in-memory
user_preferences = {}

# Logging for debugging
logging.basicConfig(level=logging.INFO)


# Load configuration
def load_config():
    try:
        with open("config.json", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: config.json not found. Please create it with your Telegram bot token.")
        exit(1)
    except json.JSONDecodeError:
        print("Error: config.json is invalid. Please check its format.")
        exit(1)


# Load model and tokenizer
def get_translator(source_lang, target_lang):
    key = (source_lang, target_lang)
    if key not in translation_models:
        return None, None
    if key not in model_cache:
        model_name = translation_models[key]
        tokenizer_cache[key] = MarianTokenizer.from_pretrained(model_name)
        model_cache[key] = MarianMTModel.from_pretrained(model_name)
    return model_cache[key], tokenizer_cache[key]


# /start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to LinguistBot!\n"
        "Use /setlang [en|uk|pl] to choose your target language.\n"
        "I’ll attach personalized Translate buttons under foreign messages."
    )


# /setlang command handler
async def set_language(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    args = context.args
    if not args or args[0] not in supported_languages:
        await update.message.reply_text("❗ Please specify a valid language: en, uk, or pl. Example: /setlang en")
        return
    user_preferences[user_id] = args[0]
    await update.message.reply_text(f"✅ Target language set to: {args[0]}")


# Add Translate buttons directly under the original message
async def translate_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    text = message.text

    if not text or message.from_user.is_bot:
        return  # Ignore empty or bot messages

    try:
        source_lang = detect(text)
    except Exception:
        return

    if source_lang not in supported_languages:
        return  # Skip if source language is not supported

    # Build a list of buttons for all unique target languages where model exists
    buttons = []
    seen_targets = set()

    for user_id, target_lang in user_preferences.items():
        if source_lang == target_lang:
            continue  # Skip if already same language
        if target_lang in seen_targets:
            continue  # Avoid duplicate buttons
        if (source_lang, target_lang) not in translation_models:
            continue  # Skip if no model available
        seen_targets.add(target_lang)

        target_name = language_names.get(target_lang, target_lang)
        buttons.append(
            [InlineKeyboardButton(
                f"🔤 Translate to {target_name}",
                callback_data=f"translate|{source_lang}|{target_lang}"
            )]
        )

    if buttons:
        keyboard = InlineKeyboardMarkup(buttons)
        await message.reply_text("Translate options:", reply_markup=keyboard)


# Handle button press (ephemeral response)
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()  # Dismiss the loading indicator

    try:
        action, source_lang, target_lang = query.data.split("|", 2)
    except ValueError:
        await query.answer("⚠️ Error processing request.", show_alert=True)
        return

    original_message = query.message.reply_to_message
    if not original_message or not original_message.text:
        await query.answer("⚠️ Original message not found.", show_alert=True)
        return

    text = original_message.text

    model, tokenizer = get_translator(source_lang, target_lang)
    if not model:
        await query.answer("⚠️ No model available for this language pair.", show_alert=True)
        return

    inputs = tokenizer([text], return_tensors="pt", padding=True, max_length=512, truncation=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    # Construct message with link if possible
    chat = query.message.chat
    message_id = original_message.message_id
    if chat.username:
        link = f"https://t.me/{chat.username}/{message_id}"
        msg = f"Original: {link}\n{text}\n\n💬 ({source_lang} → {target_lang}): {translated_text}"
    else:
        msg = f"Original: {text}\n\n💬 ({source_lang} → {target_lang}): {translated_text}"

    # Show translation ONLY to the user who clicked, in their private chat
    await context.bot.send_message(
        chat_id=query.from_user.id,
        text=msg
    )


# Set up the bot
def main():
    config = load_config()
    application = Application.builder().token(config["telegrambot_token"]).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("setlang", set_language))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, translate_message))
    application.add_handler(CallbackQueryHandler(button_handler))

    application.run_polling()


if __name__ == "__main__":
    main()