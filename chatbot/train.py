import fasttext
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. Dataset ---
# You can expand this dataset in the future to make the bot smarter!
dataset = [

    # Greeting
    ("greeting", "greeting"),
    ("hello", "greeting"),
    ("আসসালামু আলাইকুম", "greeting"),
    ("হ্যালো", "greeting"),
    ("নমস্কার", "greeting"),
    ("শুভ সকাল", "greeting"),
    ("শুভ সন্ধ্যা", "greeting"),
    ("কেমন আছেন", "greeting"),
    ("রবি সালাম", "greeting"),
    ("শুভ দিন", "greeting"),
    ("শুভ রাত্রি", "greeting"),
    ("সালাম", "greeting"),
    ("শুভেচ্ছা", "greeting"),
    ("স্বাগতম", "greeting"),


    # Balance Check
# Balance Check (Bangla only)
    ("ব্যালেন্স কত", "balance_check"),
    ("টাকা চেক", "balance_check"),
    ("একউন্টে কত টাকা", "balance_check"),
    ("ব্যালান্স চেক", "balance_check"),
    ("আমার ব্যালেন্স", "balance_check"),
    ("টাকা", "balance_check"),
    ("ব্যালেন্স", "balance_check"),
    ("টাকা কত", "balance_check"),
    ("টাকা দেখবো", "balance_check"),
    ("আমার একাউন্টে কত টাকা আছে", "balance_check"),
    ("একাউন্ট ব্যালেন্স", "balance_check"),
    ("টাকা ব্যালেন্স", "balance_check"),
    ("ব্যালেন্স জানতে চাই", "balance_check"),
    ("টাকা কত আছে", "balance_check"),
    ("ব্যালেন্স দেখাও", "balance_check"),

    # Internet Package Buy (Bangla only)
    ("ইন্টারনেট কিনবো", "internet_package_buy"),
    ("এমবি লাগবে", "internet_package_buy"),
    ("ইন্টা লাগবে", "internet_package_buy"),
    ("নেট লাগবে", "internet_package_buy"),
    ("জি বি কিনবো", "internet_package_buy"),
    ("ডাটা লাগবে", "internet_package_buy"),
    ("ইন্টারনেট প্যাকেজ", "internet_package_buy"),
    ("নেট চালু করবো", "internet_package_buy"),
    ("ইন্টারনেট চাই", "internet_package_buy"),
    ("নেট কিনবো", "internet_package_buy"),
    ("এমবি কিনব", "internet_package_buy"),
    ("ডাটা প্যাক", "internet_package_buy"),
    ("ইন্ট প্যাক", "internet_package_buy"),
    ("ডাটা কিনবো", "internet_package_buy"),
    ("ইন্টারনেট লাগবে", "internet_package_buy"),
    ("নেট প্যাক লাগবে", "internet_package_buy"),
    ("ইন্টারনেট প্যাক কিনবো", "internet_package_buy"),
    ("এমবি প্যাক লাগবে", "internet_package_buy"),
    ("ডাটা অফার লাগবে", "internet_package_buy"),

    # Internet Offer (only Bengali)
    ("ইন্টারনেট অফার", "internet_offer"),
    ("এমবি অফার", "internet_offer"),
    ("ইন্টা অফার", "internet_offer"),
    ("নেট অফার", "internet_offer"),
    ("ডাটা অফার", "internet_offer"),
    ("কি অফার আছে", "internet_offer"),
    ("ইন্ট", "internet_offer"),
    ("ইন্টার", "internet_offer"),
    ("ইন্টারন", "internet_offer"),
    ("আজকের ইন্টারনেট অফার", "internet_offer"),
    ("ইন্টারনেট এর অফার", "internet_offer"),
    ("জি বি অফার", "internet_offer"),
    ("এম বি অফার", "internet_offer"),
    ("ডাটা প্যাক অফার", "internet_offer"),

    
    # Voice Offer
    # Voice Offer (Bangla only)
    ("মিনিট লাগবে", "voice_offer"),
    ("কথা বলার অফার", "voice_offer"),
    ("টকটাইম অফার", "voice_offer"),
    ("ভয়েস প্যাকেজ", "voice_offer"),
    ("মিনিট অফার", "voice_offer"),
    ("কল অফার", "voice_offer"),
    ("কথা বলার মিনিট", "voice_offer"),
    ("ভয়েস অফার", "voice_offer"),
    ("কল রেট", "voice_offer"),
    ("মিনিট প্যাক", "voice_offer"),
    ("টকটাইম", "voice_offer"),
    ("ভয়েস প্যাক", "voice_offer"),
    ("মিনিট চাই", "voice_offer"),
    ("কল প্যাক লাগবে", "voice_offer"),
    ("ভয়েস মিনিট অফার", "voice_offer"),
    ("মিনিট কিনবো", "voice_offer"),

    
    # VAS Service
    ("গান সেট", "vas_service"),
    ("কলার টিউন", "vas_service"),
    ("গুন গুন", "vas_service"),
    ("ওয়েলকাম টিউন", "vas_service"),
    ("রিংটোন", "vas_service"),
    ("মিস কল", "vas_service"),
    ("মিসড কল অ্যালার্ট", "vas_service"),
    ("সিআরবিটি", "vas_service"),
    ("ভিএএস সার্ভিস", "vas_service"),
    ("গান লাগবে", "vas_service"),
    ("কলার টিউন চাই", "vas_service"),
    ("রিংটোন লাগবে", "vas_service"),
    ("ওয়েলকাম টিউন চাই", "vas_service"),
    ("মিসড কল সার্ভিস", "vas_service"),

    
    # Complain
    ("kono network nai", "complain_registration"),
# Complain Registration (Bangla only)
    ("অভিযোগ", "complain_registration"),
    ("নেটওয়ার্ক সমস্যা", "complain_registration"),
    ("টাকা কাটছে কেন", "complain_registration"),
    ("সমস্যা", "complain_registration"),
    ("নেট নাই", "complain_registration"),
    ("কল যাচ্ছে না", "complain_registration"),
    ("বাজে সার্ভিস", "complain_registration"),
    ("অভিযোগ করবো", "complain_registration"),
    ("নেটওয়ার্ক ভালো না", "complain_registration"),
    ("টাকা কেটে নিচ্ছে", "complain_registration"),
    ("সার্ভিস সমস্যা", "complain_registration"),
    ("কল হচ্ছে না", "complain_registration"),
    ("নেটওয়ার্ক নেই", "complain_registration"),
    ("টাকা কেটে যাচ্ছে", "complain_registration"),
    ("অভিযোগ দিতে চাই", "complain_registration"),

    
    # Confirm Yes (Bangla only)
    ("হ্যাঁ", "confirm_yes"),
    ("হ্যা", "confirm_yes"),
    ("জি", "confirm_yes"),
    ("হুম", "confirm_yes"),
    ("অবশ্যই", "confirm_yes"),
    ("ঠিক আছে", "confirm_yes"),
    ("সম্মত", "confirm_yes"),
    ("ঠিক আছে বলছি", "confirm_yes"),
    ("হ্যাঁ চাই", "confirm_yes"),
    ("হ্যাঁ লাগবে", "confirm_yes"),

    ("হ্যাঁ", "confirm_yes"),
    ("হ্যা", "confirm_yes"),
    ("জি", "confirm_yes"),
    ("হুম", "confirm_yes"),
    ("অবশ্যই", "confirm_yes"),
    ("agree", "confirm_yes"),
    
    # Confirm No (Bangla only)
    ("না", "confirm_no"),
    ("নাহ", "confirm_no"),
    ("দরকার নাই", "confirm_no"),
    ("থাক", "confirm_no"),
    ("লাগবে না", "confirm_no"),
    ("বাতিল", "confirm_no"),
    ("ক্যানসেল", "confirm_no"),
    ("প্রয়োজন নেই", "confirm_no"),
    ("চাই না", "confirm_no"),
    ("এখন দরকার নেই", "confirm_no"),
    ("না লাগবে", "confirm_no"),
    ("না চাই", "confirm_no"),

]

def train_model():
    # --- 2. Create FastText Training File ---
    train_file = "train_fasttext.txt"
    with open(train_file, "w", encoding="utf-8") as f:
        for text, intent in dataset:
            # FastText format: __label__intent_name text
            f.write(f"__label__{intent} {text.lower()}\n")
            
    # --- 3. Train the Model ---
    logger.info(f"Training fasttext intent model on {len(dataset)} samples...")
    # minn=2, maxn=5 enables character n-grams (subwords) like "ইন্টা"
    model = fasttext.train_supervised(
        input=train_file, 
        epoch=100, 
        wordNgrams=2, 
        minn=2, 
        maxn=5
    )
    
    # --- 4. Save the Model ---
    model_filename = 'intent_model.bin'
    model.save_model(model_filename)
    logger.info(f"Model saved successfully to {model_filename}")
    
    # Cleanup text file
    if os.path.exists(train_file):
        os.remove(train_file)

if __name__ == "__main__":
    train_model()
