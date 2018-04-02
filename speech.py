#!usr/bin/python3

import speech_recognition as sr
import wave
import contextlib
import pandas as pd
import matplotlib.pyplot as plt


fname = '/home/osboxes/p_projects/scene.wav'
with contextlib.closing(wave.open(fname,'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    total_length = frames / float(rate)
    total_length = int(round(total_length))
    print(total_length)

r = sr.Recognizer()
harvard = sr.AudioFile(fname)
transcript_str = []
with harvard as source:
	for t1 in range(int(total_length/10)):
		transcript_str.append(r.record(source, duration=10))
		
	transcript_str.append(r.record(source, duration=int(total_length%10)))
	
transcript_strings = []
for trs in transcript_str:
	try:
		dailogue = r.recognize_google(trs)
		transcript_strings.append(dailogue)
	except:
		pass


def transcript_word_analysis():
	word_matcher = {}
	for str_words in transcript_strings:
		words = str_words.split(" ")
		for j in words:
			j = j.lower()
			if j in word_matcher:
				word_matcher[j]+=1
			else:
				word_matcher[j]=1

	word_series = pd.Series(word_matcher,name="Word Counter")
	print(word_series)
	word_series.plot.bar()
	plt.show()


if __name__ == "__main__":
	print("Version of Speech Analysis Package is -", sr.__version__)
	print(transcript_strings)
	transcript_word_analysis()
	pass

