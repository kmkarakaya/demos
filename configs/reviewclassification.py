
def load_model():
    import tensorflow as tf 
    import pandas as pd
    from tensorflow.keras.layers import TextVectorization
    import pickle
    pkl_file = open("./configs/id_to_category.pkl", "rb")
    id_to_category = pickle.load(pkl_file)
    tr_stop_words = pd.read_csv('./configs/tr_stop_word.txt',header=None)
    vocab_size = 20000  # Only consider the top 20K words
    max_len = 50  # Maximum review (text) size in words

    @tf.keras.utils.register_keras_serializable()
    def custom_standardization(input_string):
        """ Remove html line-break tags and handle punctuation """
        no_uppercased = tf.strings.lower(input_string, encoding='utf-8')
        no_stars = tf.strings.regex_replace(no_uppercased, "\*", " ")
        no_repeats = tf.strings.regex_replace(no_stars, "devamını oku", "")    
        no_html = tf.strings.regex_replace(no_repeats, "<br />", "")
        no_digits = tf.strings.regex_replace(no_html, "\w*\d\w*","")
        #no_punctuations = tf.strings.regex_replace(no_digits, f"([{tf.string.punctuation}])", r" ")
        #remove stop words
        no_stop_words = ' '+no_digits+ ' '
        
        for each in tr_stop_words.values:
            no_stop_words = tf.strings.regex_replace(no_stop_words, ' '+each[0]+' ' , r" ")
        no_extra_space = tf.strings.regex_replace(no_stop_words, " +"," ")
        #remove Turkish chars
        no_I = tf.strings.regex_replace(no_extra_space, "ı","i")
        no_O = tf.strings.regex_replace(no_I, "ö","o")
        no_C = tf.strings.regex_replace(no_O, "ç","c")
        no_S = tf.strings.regex_replace(no_C, "ş","s")
        no_G = tf.strings.regex_replace(no_S, "ğ","g")
        no_U = tf.strings.regex_replace(no_G, "ü","u")

        return no_U

    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=vocab_size+2,
        output_mode="int",
        output_sequence_length=max_len,)

    
    loaded_end_to_end_model = tf.keras.models.load_model(".\models\end_to_end_model")
    return loaded_end_to_end_model, id_to_category

