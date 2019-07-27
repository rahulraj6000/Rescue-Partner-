import tweepy 

# Fill the X's with the credentials obtained by 
# following the above mentioned procedure. 
consumer_key = "4balYj##################### "
consumer_secret = "############################"
access_key = "############################################"
access_secret = "###################################################"

# Function to extract tweets 
def get_tweets(username): 
		
		# Authorization to consumer key and consumer secret 
		auth = tweepy.OAuthHandler(consumer_key, consumer_secret) 

		# Access to user's access key and access secret 
		auth.set_access_token(access_key, access_secret) 

		# Calling api 
		api = tweepy.API(auth) 

		# 200 tweets to be extracted 
		number_of_tweets=200
		tweets = api.search("Trump")##api.user_timeline(screen_name=username) 

		# Empty Array 
		tmp=[] 

		# create array of tweet information: username, 
		# tweet id, date/time, text 
		#tweets_for_csv = [tweet.text for tweet in tweets] # CSV file created 
		for j in tweets: 

			# Appending tweets to the empty array tmp 
	         print(j)
         
		# Printing the tweets 
		#print(tmp) 

