import re

from tqdm import tqdm


def make_sentences(batch, filename):
    def recursive_split(i, listed):
        if i >= len(listed):
            return listed

        if len(listed[i]) > 500:
            j = listed[i][:500].rindex(" ")
            listed = (
                listed[:i] + [listed[i][:j]] + [listed[i][j + 1 :]] + listed[i + 2 :]
            )
            listed = recursive_split(i + 1, listed)
        else:
            listed = recursive_split(i + 1, listed)

        return listed

    text = ""
    for b in batch:
        text += "".join(b)

    x_s = re.split(
        r"((?<!www|WWW|[A-Za-z0-9]{2}/)(?<![\s.?!;(][A-Za-z0-9])[;.?!](?![0-9])(?!\s*[;.!?])(?![A-Za-z]*/))",
        text,
    )

    single_sentences_list = [x.strip() for x in x_s if x.strip() != ""]
    true_sentences = [
        single_sentences_list[i] + single_sentences_list[i + 1]
        for i in range(0, len(single_sentences_list) - 1, 2)
    ]

    if len(single_sentences_list) % 2 != 0:
        true_sentences.append(single_sentences_list[-1])
    print("single sentences done", true_sentences[:5])

    updated_sentences = []
    j = 0
    for i in range(0, len(true_sentences)):
        if len(true_sentences[i]) < 16 and i != len(true_sentences) - 1:
            true_sentences[i + 1] = true_sentences[i] + true_sentences[i + 1]
        elif i == len(true_sentences) - 1:
            updated_sentences.append(true_sentences[i])
        else:
            updated_sentences.append(true_sentences[i])

    new_sentences = updated_sentences
    sentences = []
    for i, sentence in enumerate(new_sentences):
        sentence = re.sub(
            '\\n|\\t|\\r|\\"',
            " ",
            re.sub(r"\s[\s]+", "", re.sub(r"[\n\t\r]+", " ", sentence)),
        )
        sentences.append({"sentence": sentence, "index": i, "filename": filename})

    buffer_size = 1
    for i in tqdm(range(len(sentences)), desc="Creating combined sentences"):
        combined_sent = ""
        for j in range(i - buffer_size, i):
            if j >= 0:
                combined_sent += sentences[j]["sentence"] + " "

        combined_sent += sentences[i]["sentence"]
        sentences[i]["combined_sent"] = combined_sent

    return sentences
