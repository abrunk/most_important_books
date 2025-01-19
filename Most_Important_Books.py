# After experimenting with reticulate in RStudio, I decided to switch to doing this in PyCharm for simplicity

''' Most Important Books

I am an avid reader, and have been since I was a small child. My interests include historical fiction, fantasy/sci-fi, biographies, histories, and works of political philosophy.

In the last few years however, I've been attempting to read more books that are considered 'classics.' There is no universal definition of what is a 'classic' though there are many good lists such as the 'Great Books of the Western World' prepared by the late great Mortimer Adler.

A few examples of 'classics' I've read in the last few years are:

* Meditations by Marcus Aurelius
* The Power and the Glory by Graham Greene
* All Quiet on the Western Front by Erich Maria Remarque
* The Stranger by Albert Camus
* Utopia by Thomas More
* The Rights of Man by Thomas Paine
* Moby Dick by Herman Melville
* Brave New World by Aldous Huxley
* Narrative of Life of Frederick Douglass by Frederick Douglass
* The Time Machine by HG Wells
* Heart of Darkness by Joseph Conrad

These are all books that show up on various lists of 'classics' from different eras and genres. But any such decisions are arbitrary. Who decides what a classic is? How should one determine which are worth reading? How to make decisions about which books to read given finite time to read them?
'''

# The Concept - A Model to Determine the Importance of Classics
# I turned to Project Gutenberg, which has an extensive library of public domain books. There is an existing Python library called gutenbergpy that has been designed to make it easier to download Project Gutenberg texts into Python code.
# As an example, here is a full printing of 'Moby Dick' without header information using the package instructions available here: https://github.com/raduangelescu/gutenbergpy and the unique identifier for the Moby Dick text found by googling here https://www.gutenberg.org/ebooks/15

import gutenbergpy.textget
import collections
import numpy as np
import pandas as pd
import nltk
import re
import random
import bs4 as bs
import urllib.request

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab', quiet=True)
nltk.download('words', quiet=True)

# return Moby Dick text using Project Gutenberg ID
moby_dick = gutenbergpy.textget.get_text_by_id(15)

# print(moby_dick)
clean_book  = gutenbergpy.textget.strip_headers(moby_dick)

# Try to decode the text
try:
    string_book = clean_book.decode('utf-8')
except UnicodeDecodeError:
    string_book = clean_book.decode('latin-1')  # fallback encoding

# We can then create a corpus and tokenize the text for NLP processing as follows:

corpus = nltk.sent_tokenize(string_book)

for i in range(len(corpus)):
    corpus[i] = corpus[i].lower()
    corpus[i] = re.sub(r'\W', ' ', corpus[i])
    corpus[i] = re.sub(r'\s+', ' ', corpus[i])

wordfreq = {}
for sentence in corpus:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1

# In addition to getting the text of any individual book in Project Gutenberg, we can also connect to their database of book metadata information:

from gutenbergpy.gutenbergcache import GutenbergCache
#for sqlite
GutenbergCache.create()

# Now, let's connect to the database and start some querys.

cache  = GutenbergCache.get_cache()
my_cursor = cache.native_query("SELECT a.id,b.name,c.name FROM books a inner join titles b on a.id = b.id inner join authors c on a.id = c.id where a.id = 15")
for row in my_cursor:
    print(row)

# For this step, we will rely on some existing code available here
# https://stackoverflow.com/questions/20290870/improving-the-extraction-of-human-names-with-nltk
# with some minor tweaks to create a list of names in the text of Moby Dick.

token_list = []

def extract_entities(text):
  for sent in nltk.sent_tokenize(text):
    for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
      if hasattr(chunk, 'label'):
        # print(chunk.label(), ' '.join(c[0] for c in chunk.leaves()))
        for c in chunk.leaves():
          this_row = c[0]
          token_list.append(this_row)

extract_entities(string_book)

print(token_list)

from collections import Counter
Counter(token_list)

# From here, the next step is to remove common words that you would find in the dictionary
# to ensure that we are working with a list of unique names that could be identified as a
# reference to the text. First let's try the nltk words list.

from nltk.corpus import words

"would" in words.words()
"Starbuck" in words.words()

# This doesn't seem to work, as the list of words contains names and not just English dictionary terms. Let's try referencing an actual dictionary.

# import enchant
#d = enchant.Dict("en_US")
# d.check("would")
# d.check("Starbuck")
# d.check("Etymology")
# d.check("etymology")
# d.check("Ahab")
# d.check("ahab")

# Enchant doesn't seem to work no matter what I do, so let's try it with textblob instead per Claude's suggestion.

from textblob import Word

# Check words
word1 = Word("would")
word2 = Word("Starbuck")
word3 = Word("Etymology")
word4 = Word("etymology")
word5 = Word("Ahab")
word6 = Word("ahab")

print(word1.correct())  # Prints the word if correct, or suggests correction
print(word2.correct())
print(word3.correct())
print(word4.correct())
print(word5.correct())
print(word6.correct())

# That doesn't seem to do what I need it to do. Let's try it with nltk which I'm already using

# Test examples
from nltk import corpus

# Download required datasets (only need to do this once if you haven't already)
nltk.download('words')
nltk.download('names')

def check_word_type(word):
    word_lower = word.lower()

    # Get standard dictionary words
    word_list = set(corpus.words.words())

    # Get common first names
    common_names = set(name.lower() for name in corpus.names.words())

    # Additional check for proper nouns (capitalized words not in dictionary)
    is_proper_noun = word[0].isupper() and word_lower not in word_list

    is_dict_word = word_lower in word_list
    is_common_name = word_lower in common_names

    return (is_common_name or is_proper_noun) and not (is_dict_word)


# Test examples
test_words = ['would', 'Starbuck', 'Etymology', 'etymology', 'Ahab', 'ahab','Stubb','Whale','Captain','Sperm','Flask','Dick','Pip','God','Shadrach']
for word in test_words:
    result = check_word_type(word)
    print(word + " : " + str(result))

# A random word like 'would' returns 'FALSE' while 'Starbuck' - a name of a character in Moby Dick - returns 'TRUE'. There's also some random junk in here, so let's first remove punctuation
# Note: I had previously converted everything here to all lowercase but realized that case information is useful for determining names so commented out that code

import string
from collections import Counter

# token_list = [item.lower() for item in token_list]
for index, item  in enumerate(token_list):
  print(index,item)
  token_list[index] = item.translate(str.maketrans('','',string.punctuation))
#  if d.check(item):
#    token_list.pop(index)
print(token_list)

# Next, iterate through again to remove everything that's an English dictionary word.
# Note that I tried combining these steps to enumerate through the list only once, but the code didn't seem to work correctly.

mod_token_list = [item for item in token_list if check_word_type(item)]

print("After removing dictionary words:", mod_token_list)

# Count the remaining tokens
name_counts = Counter(mod_token_list)
print("Final name counts:", name_counts)

# This didn't turn out perfectly, for example I lost the name 'Stubb' which appears numerous times in the text.
# One idea I have for improving this name checker is to see how often a word appears as a proper noun. If it is always capitalized then
# the model could assess that it is a name even if it appears in the dictionary. But then I might also end up with words like 'Captain'

# Now that I have a working list of names from Moby Dick, my next step is to check it against the titles of all the other books in Project Gutenberg

cache = GutenbergCache.get_cache()
# Get all titles and authors with book IDs
my_cursor = cache.native_query("""
    SELECT DISTINCT 
        a.id,
        b.name as title,
        c.name as author
    FROM books a 
    INNER JOIN titles b on a.id = b.id 
    INNER JOIN authors c on a.id = c.id
""")

# Create dictionaries to store matches with their details
title_references = {}  # Will store {matched_word: [(book_id, full_title, count)]}
author_references = {}  # Will store {matched_word: [(book_id, full_author, count)]}

# Convert database results while keeping track of full information
title_lookup = {}  # {lowercase_word: [(book_id, full_title)]}
author_lookup = {}  # {lowercase_word: [(book_id, full_author)]}

for row in my_cursor:
    book_id, full_title, full_author = row

    # Process title
    title_words = full_title.translate(str.maketrans('', '', string.punctuation)).split()
    for word in title_words:
        word_lower = word.lower()
        if word_lower not in title_lookup:
            title_lookup[word_lower] = []
        title_lookup[word_lower].append((book_id, full_title))

    # Also add full title
    title_lower = full_title.lower()
    if title_lower not in title_lookup:
        title_lookup[title_lower] = []
    title_lookup[title_lower].append((book_id, full_title))

    # Process author
    author_words = full_author.translate(str.maketrans('', '', string.punctuation)).split()
    for word in author_words:
        word_lower = word.lower()
        if word_lower not in author_lookup:
            author_lookup[word_lower] = []
        author_lookup[word_lower].append((book_id, full_author))

    # Also add full author name
    author_lower = full_author.lower()
    if author_lower not in author_lookup:
        author_lookup[author_lower] = []
    author_lookup[author_lower].append((book_id, full_author))

# Check matches from mod_token_list
for name in mod_token_list:
    name_lower = name.lower()

    # Check for title matches
    if name_lower in title_lookup:
        for book_id, full_title in title_lookup[name_lower]:
            key = (book_id, full_title)
            title_references[key] = title_references.get(key, 0) + 1

    # Check for author matches
    if name_lower in author_lookup:
        for book_id, full_author in author_lookup[name_lower]:
            key = (book_id, full_author)
            author_references[key] = author_references.get(key, 0) + 1

# Sort by frequency
sorted_title_refs = sorted(title_references.items(), key=lambda x: x[1], reverse=True)
sorted_author_refs = sorted(author_references.items(), key=lambda x: x[1], reverse=True)

# Print results
print("\nMost referenced book titles in Moby Dick:")
print("Book ID | Count | Title")
print("-" * 50)
for (book_id, title), count in sorted_title_refs[:30]:
    print(f"#{book_id:<7} | {count:<5} | {title}")

print("\nMost referenced authors in Moby Dick:")
print("Book ID | Count | Author")
print("-" * 50)
for (book_id, author), count in sorted_author_refs[:30]:
    print(f"#{book_id:<7} | {count:<5} | {author}")

# When I run this, its clear that we have an issue at present with the model referencing more obscure sources that are found in Project Gutenberg instead of
# just the most important works that are available there. This presents a design issue: if I filter out texts based on some criteria I risk devaluing the model
# by excluding lesser known texts that may be very influential. But if I keep things like the Aleutian dictionary in then the logic doesn't work super well

# Let's see how long these lists are:.
cache = GutenbergCache.get_cache()

# Count distinct titles
my_cursor = cache.native_query("""
    SELECT COUNT(DISTINCT b.name) as title_count
    FROM books a 
    INNER JOIN titles b on a.id = b.id 
""")
for row in my_cursor:
    title_count = row[0]
    print(f"Total number of unique titles: {title_count}")

# Count distinct authors
my_cursor = cache.native_query("""
    SELECT COUNT(DISTINCT c.name) as author_count
    FROM books a 
    INNER JOIN authors c on a.id = c.id
""")
for row in my_cursor:
    author_count = row[0]
    print(f"Total number of unique authors: {author_count}")

# Let's look at the data structure
my_cursor = cache.native_query("""
    SELECT a.id, 
           COUNT(DISTINCT b.name) as title_count,
           COUNT(DISTINCT c.name) as author_count
    FROM books a 
    LEFT JOIN titles b on a.id = b.id 
    LEFT JOIN authors c on a.id = c.id
    GROUP BY a.id
    HAVING title_count != author_count
    LIMIT 5
""")

print("\nSample of books with different title/author counts:")
print("Book ID | Titles | Authors")
print("-" * 35)
for row in my_cursor:
    print(f"#{row[0]:<7} | {row[1]:<6} | {row[2]}")

# Ok, we now know that there are 67,860 titles and 47,627 authors present in Project Gutenberg.
# As a proxy for the importance of the text, we could use the number of downloads from Project Gutenberg. Let's see if that column exists in the data.

cache = GutenbergCache.get_cache()
# Check table structure
my_cursor = cache.native_query("""
SELECT sql 
FROM sqlite_master 
WHERE type='table' AND name='books'
""")

for row in my_cursor:
    print("Books table structure:")
    print(row[0])

# It looks like there's a 'numdownloads' column that would allow us to filter out very obscure books.

cache = GutenbergCache.get_cache()

# Get top 2000 most downloaded books and their titles/authors
my_cursor = cache.native_query("""
   SELECT DISTINCT 
       a.id,
       b.name as title,
       c.name as author,
       a.numdownloads
   FROM books a 
   INNER JOIN titles b on a.id = b.id 
   INNER JOIN authors c on a.id = c.id
   ORDER BY a.numdownloads DESC
   LIMIT 2000
""")

# Create dictionaries to store matches with their details and download counts
title_references = {}  # Will store {(book_id, full_title): (count, downloads)}
author_references = {}  # Will store {(book_id, full_author): (count, downloads)}

# Convert database results while keeping track of full information
title_lookup = {}  # {lowercase_word: [(book_id, full_title, downloads)]}
author_lookup = {}  # {lowercase_word: [(book_id, full_author, downloads)]}

for row in my_cursor:
    book_id, full_title, full_author, downloads = row

    # Process title
    title_words = full_title.translate(str.maketrans('', '', string.punctuation)).split()
    for word in title_words:
        word_lower = word.lower()
        if word_lower not in title_lookup:
            title_lookup[word_lower] = []
        title_lookup[word_lower].append((book_id, full_title, downloads))

    # Process author
    author_words = full_author.translate(str.maketrans('', '', string.punctuation)).split()
    for word in author_words:
        word_lower = word.lower()
        if word_lower not in author_lookup:
            author_lookup[word_lower] = []
        author_lookup[word_lower].append((book_id, full_author, downloads))

# Check matches from mod_token_list
for name in mod_token_list:
    name_lower = name.lower()

    # Check for title matches
    if name_lower in title_lookup:
        for book_id, full_title, downloads in title_lookup[name_lower]:
            key = (book_id, full_title)
            if key not in title_references:
                title_references[key] = [0, downloads]  # [count, downloads]
            title_references[key][0] += 1

    # Check for author matches
    if name_lower in author_lookup:
        for book_id, full_author, downloads in author_lookup[name_lower]:
            key = (book_id, full_author)
            if key not in author_references:
                author_references[key] = [0, downloads]  # [count, downloads]
            author_references[key][0] += 1

# Sort by frequency
sorted_title_refs = sorted(title_references.items(), key=lambda x: x[1][0], reverse=True)
sorted_author_refs = sorted(author_references.items(), key=lambda x: x[1][0], reverse=True)

# Print results
print("\nMost referenced book titles in Moby Dick (from top 2000 most downloaded books):")
print("Book ID | References | Downloads | Title")
print("-" * 70)
for (book_id, title), (count, downloads) in sorted_title_refs[:10]:
    print(f"#{book_id:<7} | {count:<10} | {downloads:<9} | {title}")

print("\nMost referenced authors in Moby Dick (from top 2000 most downloaded books):")
print("Book ID | References | Downloads | Author")
print("-" * 70)
for (book_id, author), (count, downloads) in sorted_author_refs[:10]:
    print(f"#{book_id:<7} | {count:<10} | {downloads:<9} | {author}")

# This didn't work terribly well either, we still have lots of obscure titles referenced. The issue appears to be that the most
# important and influential books don't correspond well with which ones are most often downloaded from Project Gutenberg.

# Let's look at the distribution of downloads

cache = GutenbergCache.get_cache()

my_cursor = cache.native_query("""
    SELECT DISTINCT 
        a.id,
        b.name as title,
        c.name as author,
        a.numdownloads
    FROM books a 
    INNER JOIN titles b on a.id = b.id 
    INNER JOIN authors c on a.id = c.id
    ORDER BY a.numdownloads DESC
    LIMIT 10
""")

print("Top 10 most downloaded books:")
print("Book ID | Downloads | Title | Author")
print("-" * 70)
for row in my_cursor:
    print(f"#{row[0]:<7} | {row[3]:<9} | {row[1]} | {row[2]}")

# Ok, this isn't working either. We need a curated list to start from somehow and be able to match it to the numerical IDs in Project Gutenberg
# I pulled the top 500 books from the website 'The Greatest Books' and downloaded the csv

# Let's pull it in and filter for books that are in the public domain (published 1923 or earlier)
# We will use polars since its faster than pandas and I intend to use it for later pieces of this project

import polars as pl
greatest_books = pl.read_csv('greatest_1000_books.csv')

with pl.Config(tbl_cols=-1):
    print(greatest_books)

# Remove books published after 1923
greatest_books = greatest_books.filter(pl.col("Year") < 1924)

# Look at the list again
with pl.Config(tbl_cols=-1):
    print(greatest_books)

# That leaves us with 963 books which is a good sample to compare against.

