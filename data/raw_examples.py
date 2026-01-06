# Training Examples for Music Application
# Auto-generated expanded dataset with 25-30 examples per function
# Format: (user_input, function_name, arguments_dict)

TRAINING_EXAMPLES = [
    ("Play Bohemian Rhapsody", "play_song",
     {'song_name': 'Bohemian Rhapsody'}),

    ("Play Blinding Lights by The Weeknd", "play_song",
     {'song_name': 'Blinding Lights', 'artist': 'The Weeknd'}),

    ("Play Shape of You", "play_song",
     {'song_name': 'Shape of You'}),

    ("Play Billie Jean by Michael Jackson", "play_song",
     {'song_name': 'Billie Jean', 'artist': 'Michael Jackson'}),

    ("Play Hotel California from Eagles", "play_song",
     {'song_name': 'Hotel California', 'artist': 'Eagles'}),

    ("Play Wonderwall by Oasis", "play_song",
     {'song_name': 'Wonderwall', 'artist': 'Oasis'}),

    ("Play Imagine by John Lennon", "play_song",
     {'song_name': 'Imagine', 'artist': 'John Lennon'}),

    ("Play Bad Guy by Billie Eilish", "play_song",
     {'song_name': 'Bad Guy', 'artist': 'Billie Eilish'}),

    ("Play Stairway to Heaven", "play_song",
     {'song_name': 'Stairway to Heaven'}),

    ("Play Smells Like Teen Spirit", "play_song",
     {'song_name': 'Smells Like Teen Spirit'}),

    ("Can you play Levitating", "play_song",
     {'song_name': 'Levitating'}),

    ("I want to hear Shallow", "play_song",
     {'song_name': 'Shallow'}),

    ("Put on Radioactive", "play_song",
     {'song_name': 'Radioactive'}),

    ("Start playing Rolling in the Deep", "play_song",
     {'song_name': 'Rolling in the Deep'}),

    ("Play Believer by Imagine Dragons", "play_song",
     {'song_name': 'Believer', 'artist': 'Imagine Dragons'}),

    ("Play Circles by Post Malone", "play_song",
     {'song_name': 'Circles', 'artist': 'Post Malone'}),

    ("Play Watermelon Sugar", "play_song",
     {'song_name': 'Watermelon Sugar'}),

    ("Play Peaches by Justin Bieber", "play_song",
     {'song_name': 'Peaches', 'artist': 'Justin Bieber'}),

    ("Play good 4 u", "play_song",
     {'song_name': 'good 4 u'}),

    ("Play Drivers License", "play_song",
     {'song_name': 'Drivers License'}),

    ("Play Montero", "play_song",
     {'song_name': 'Montero'}),

    ("Play Stay by The Kid LAROI", "play_song",
     {'song_name': 'Stay', 'artist': 'The Kid LAROI'}),

    ("Play Heat Waves", "play_song",
     {'song_name': 'Heat Waves'}),

    ("Play Someone Like You by Adele", "play_song",
     {'song_name': 'Someone Like You', 'artist': 'Adele'}),

    ("Play Hello from Adele", "play_song",
     {'song_name': 'Hello', 'artist': 'Adele'}),

    ("Play Perfect by Ed Sheeran", "play_song",
     {'song_name': 'Perfect', 'artist': 'Ed Sheeran'}),

    ("Play Happier by Marshmello", "play_song",
     {'song_name': 'Happier', 'artist': 'Marshmello'}),

    ("Play Roses by SAINt JHN", "play_song",
     {'song_name': 'Roses', 'artist': 'SAINt JHN'}),

    ("Play Roxanne", "play_song",
     {'song_name': 'Roxanne'}),

    ("Play Dance Monkey", "play_song",
     {'song_name': 'Dance Monkey'}),

    ("Pause the music", "playback_control",
     {'action': 'pause'}),

    ("Skip this song", "playback_control",
     {'action': 'skip'}),

    ("Next song", "playback_control",
     {'action': 'next'}),

    ("Go back", "playback_control",
     {'action': 'previous'}),

    ("Previous track", "playback_control",
     {'action': 'previous'}),

    ("Resume", "playback_control",
     {'action': 'resume'}),

    ("Stop playing", "playback_control",
     {'action': 'stop'}),

    ("Pause", "playback_control",
     {'action': 'pause'}),

    ("Skip", "playback_control",
     {'action': 'skip'}),

    ("Play", "playback_control",
     {'action': 'play'}),

    ("Continue playing", "playback_control",
     {'action': 'resume'}),

    ("Go to next song", "playback_control",
     {'action': 'next'}),

    ("Go to previous", "playback_control",
     {'action': 'previous'}),

    ("Stop the music", "playback_control",
     {'action': 'stop'}),

    ("Pause this", "playback_control",
     {'action': 'pause'}),

    ("Skip to next", "playback_control",
     {'action': 'skip'}),

    ("Previous song please", "playback_control",
     {'action': 'previous'}),

    ("Resume playback", "playback_control",
     {'action': 'resume'}),

    ("Stop", "playback_control",
     {'action': 'stop'}),

    ("Next track", "playback_control",
     {'action': 'next'}),

    ("Pause it", "playback_control",
     {'action': 'pause'}),

    ("Skip it", "playback_control",
     {'action': 'skip'}),

    ("Keep playing", "playback_control",
     {'action': 'resume'}),

    ("Go back to previous", "playback_control",
     {'action': 'previous'}),

    ("Stop music", "playback_control",
     {'action': 'stop'}),

    ("Next", "playback_control",
     {'action': 'next'}),

    ("Previous", "playback_control",
     {'action': 'previous'}),

    ("Skip this track", "playback_control",
     {'action': 'skip'}),

    ("Pause music", "playback_control",
     {'action': 'pause'}),

    ("Resume music", "playback_control",
     {'action': 'resume'}),

    ("Turn up the volume", "set_volume",
     {'action': 'increase'}),

    ("Turn down the volume", "set_volume",
     {'action': 'decrease'}),

    ("Set volume to 50", "set_volume",
     {'action': 'set', 'level': 50}),

    ("Volume up", "set_volume",
     {'action': 'increase'}),

    ("Volume down", "set_volume",
     {'action': 'decrease'}),

    ("Mute", "set_volume",
     {'action': 'mute'}),

    ("Unmute", "set_volume",
     {'action': 'unmute'}),

    ("Set volume to 80", "set_volume",
     {'action': 'set', 'level': 80}),

    ("Louder", "set_volume",
     {'action': 'increase'}),

    ("Quieter", "set_volume",
     {'action': 'decrease'}),

    ("Turn it up", "set_volume",
     {'action': 'increase'}),

    ("Turn it down", "set_volume",
     {'action': 'decrease'}),

    ("Make it louder", "set_volume",
     {'action': 'increase'}),

    ("Make it quieter", "set_volume",
     {'action': 'decrease'}),

    ("Increase volume", "set_volume",
     {'action': 'increase'}),

    ("Decrease volume", "set_volume",
     {'action': 'decrease'}),

    ("Set volume 70", "set_volume",
     {'action': 'set', 'level': 70}),

    ("Volume to 100", "set_volume",
     {'action': 'set', 'level': 100}),

    ("Volume to 30", "set_volume",
     {'action': 'set', 'level': 30}),

    ("Mute the music", "set_volume",
     {'action': 'mute'}),

    ("Unmute it", "set_volume",
     {'action': 'unmute'}),

    ("Turn off sound", "set_volume",
     {'action': 'mute'}),

    ("Turn on sound", "set_volume",
     {'action': 'unmute'}),

    ("Lower the volume", "set_volume",
     {'action': 'decrease'}),

    ("Raise the volume", "set_volume",
     {'action': 'increase'}),

    ("Volume 60", "set_volume",
     {'action': 'set', 'level': 60}),

    ("Max volume", "set_volume",
     {'action': 'set', 'level': 100}),

    ("Minimum volume", "set_volume",
     {'action': 'set', 'level': 0}),

    ("Boost volume", "set_volume",
     {'action': 'increase'}),

    ("Reduce volume", "set_volume",
     {'action': 'decrease'}),

    ("Create a playlist called Workout Mix", "create_playlist",
     {'name': 'Workout Mix'}),

    ("Make a new playlist named Chill Vibes", "create_playlist",
     {'name': 'Chill Vibes'}),

    ("Create playlist Party Hits", "create_playlist",
     {'name': 'Party Hits'}),

    ("New playlist called Road Trip", "create_playlist",
     {'name': 'Road Trip'}),

    ("Make a playlist called Study Music", "create_playlist",
     {'name': 'Study Music'}),

    ("Create Morning Jams playlist", "create_playlist",
     {'name': 'Morning Jams'}),

    ("Create a new playlist Favorites", "create_playlist",
     {'name': 'Favorites'}),

    ("Make playlist called Summer 2024", "create_playlist",
     {'name': 'Summer 2024'}),

    ("Create Running Mix playlist", "create_playlist",
     {'name': 'Running Mix'}),

    ("New playlist named Relaxing", "create_playlist",
     {'name': 'Relaxing'}),

    ("Create playlist My Top Songs", "create_playlist",
     {'name': 'My Top Songs'}),

    ("Make a Dance Party playlist", "create_playlist",
     {'name': 'Dance Party'}),

    ("Create Gym Workout playlist", "create_playlist",
     {'name': 'Gym Workout'}),

    ("New playlist Feel Good", "create_playlist",
     {'name': 'Feel Good'}),

    ("Create playlist called Focus", "create_playlist",
     {'name': 'Focus'}),

    ("Make playlist Weekend Vibes", "create_playlist",
     {'name': 'Weekend Vibes'}),

    ("Create Late Night playlist", "create_playlist",
     {'name': 'Late Night'}),

    ("New playlist Throwbacks", "create_playlist",
     {'name': 'Throwbacks'}),

    ("Create playlist called Mood Booster", "create_playlist",
     {'name': 'Mood Booster'}),

    ("Make Acoustic playlist", "create_playlist",
     {'name': 'Acoustic'}),

    ("Create playlist Energy Boost", "create_playlist",
     {'name': 'Energy Boost'}),

    ("New playlist Calm Down", "create_playlist",
     {'name': 'Calm Down'}),

    ("Create 90s Hits playlist", "create_playlist",
     {'name': '90s Hits'}),

    ("Make Rock Classics playlist", "create_playlist",
     {'name': 'Rock Classics'}),

    ("Create playlist Sunday Morning", "create_playlist",
     {'name': 'Sunday Morning'}),

    ("Add this to Workout Mix", "add_to_playlist",
     {'playlist_name': 'Workout Mix', 'song_name': 'this song'}),

    ("Add this song to my favorites", "add_to_playlist",
     {'playlist_name': 'favorites', 'song_name': 'this song'}),

    ("Add to Chill Vibes", "add_to_playlist",
     {'playlist_name': 'Chill Vibes', 'song_name': 'this song'}),

    ("Put this in Party Hits", "add_to_playlist",
     {'playlist_name': 'Party Hits', 'song_name': 'this song'}),

    ("Add to Road Trip playlist", "add_to_playlist",
     {'playlist_name': 'Road Trip', 'song_name': 'this song'}),

    ("Add this to Study Music", "add_to_playlist",
     {'playlist_name': 'Study Music', 'song_name': 'this song'}),

    ("Put this song in Morning Jams", "add_to_playlist",
     {'playlist_name': 'Morning Jams', 'song_name': 'this song'}),

    ("Add to Summer 2024", "add_to_playlist",
     {'playlist_name': 'Summer 2024', 'song_name': 'this song'}),

    ("Add this to Running Mix", "add_to_playlist",
     {'playlist_name': 'Running Mix', 'song_name': 'this song'}),

    ("Put in Relaxing playlist", "add_to_playlist",
     {'playlist_name': 'Relaxing', 'song_name': 'this song'}),

    ("Add to My Top Songs", "add_to_playlist",
     {'playlist_name': 'My Top Songs', 'song_name': 'this song'}),

    ("Add this to Dance Party", "add_to_playlist",
     {'playlist_name': 'Dance Party', 'song_name': 'this song'}),

    ("Put in Gym Workout", "add_to_playlist",
     {'playlist_name': 'Gym Workout', 'song_name': 'this song'}),

    ("Add to Feel Good playlist", "add_to_playlist",
     {'playlist_name': 'Feel Good', 'song_name': 'this song'}),

    ("Add this to Focus", "add_to_playlist",
     {'playlist_name': 'Focus', 'song_name': 'this song'}),

    ("Put this in Weekend Vibes", "add_to_playlist",
     {'playlist_name': 'Weekend Vibes', 'song_name': 'this song'}),

    ("Add to Late Night", "add_to_playlist",
     {'playlist_name': 'Late Night', 'song_name': 'this song'}),

    ("Add this to Throwbacks", "add_to_playlist",
     {'playlist_name': 'Throwbacks', 'song_name': 'this song'}),

    ("Put in Mood Booster", "add_to_playlist",
     {'playlist_name': 'Mood Booster', 'song_name': 'this song'}),

    ("Add to Acoustic playlist", "add_to_playlist",
     {'playlist_name': 'Acoustic', 'song_name': 'this song'}),

    ("Add this to Energy Boost", "add_to_playlist",
     {'playlist_name': 'Energy Boost', 'song_name': 'this song'}),

    ("Put in Calm Down", "add_to_playlist",
     {'playlist_name': 'Calm Down', 'song_name': 'this song'}),

    ("Add to 90s Hits", "add_to_playlist",
     {'playlist_name': '90s Hits', 'song_name': 'this song'}),

    ("Add this song to Rock Classics", "add_to_playlist",
     {'playlist_name': 'Rock Classics', 'song_name': 'this song'}),

    ("Put in Sunday Morning playlist", "add_to_playlist",
     {'playlist_name': 'Sunday Morning', 'song_name': 'this song'}),

    ("Search for Taylor Swift", "search_music",
     {'query': 'Taylor Swift'}),

    ("Find songs by Drake", "search_music",
     {'query': 'Drake'}),

    ("Search for pop music", "search_music",
     {'query': 'pop music'}),

    ("Look for Coldplay", "search_music",
     {'query': 'Coldplay'}),

    ("Search Dua Lipa", "search_music",
     {'query': 'Dua Lipa'}),

    ("Find Ariana Grande", "search_music",
     {'query': 'Ariana Grande'}),

    ("Search for The Weeknd", "search_music",
     {'query': 'The Weeknd'}),

    ("Look up Ed Sheeran", "search_music",
     {'query': 'Ed Sheeran'}),

    ("Search Bruno Mars", "search_music",
     {'query': 'Bruno Mars'}),

    ("Find Post Malone", "search_music",
     {'query': 'Post Malone'}),

    ("Search for Billie Eilish", "search_music",
     {'query': 'Billie Eilish'}),

    ("Look for rock songs", "search_music",
     {'query': 'rock songs'}),

    ("Search for workout music", "search_music",
     {'query': 'workout music'}),

    ("Find study music", "search_music",
     {'query': 'study music'}),

    ("Search for jazz", "search_music",
     {'query': 'jazz'}),

    ("Look for classical music", "search_music",
     {'query': 'classical music'}),

    ("Search hip hop", "search_music",
     {'query': 'hip hop'}),

    ("Find country music", "search_music",
     {'query': 'country music'}),

    ("Search for 80s music", "search_music",
     {'query': '80s music'}),

    ("Look for romantic songs", "search_music",
     {'query': 'romantic songs'}),

    ("Search for happy songs", "search_music",
     {'query': 'happy songs'}),

    ("Find sad songs", "search_music",
     {'query': 'sad songs'}),

    ("Search for party music", "search_music",
     {'query': 'party music'}),

    ("Look for relaxing music", "search_music",
     {'query': 'relaxing music'}),

    ("Search acoustic songs", "search_music",
     {'query': 'acoustic songs'}),

    ("Find dance music", "search_music",
     {'query': 'dance music'}),

    ("Search for indie music", "search_music",
     {'query': 'indie music'}),

    ("Look for electronic music", "search_music",
     {'query': 'electronic music'}),

    ("Search for R&B", "search_music",
     {'query': 'R&B'}),

    ("Find rap music", "search_music",
     {'query': 'rap music'}),

    ("Add this to my favorites", "add_to_favorites",
     {}),

    ("Like this song", "add_to_favorites",
     {}),

    ("Add to favorites", "add_to_favorites",
     {}),

    ("Save this song", "add_to_favorites",
     {}),

    ("Add to liked songs", "add_to_favorites",
     {}),

    ("Favorite this", "add_to_favorites",
     {}),

    ("I love this song", "add_to_favorites",
     {}),

    ("Save this to favorites", "add_to_favorites",
     {}),

    ("Like this", "add_to_favorites",
     {}),

    ("Add this to liked", "add_to_favorites",
     {}),

    ("Favorite this song", "add_to_favorites",
     {}),

    ("Save it", "add_to_favorites",
     {}),

    ("Like it", "add_to_favorites",
     {}),

    ("Add to my likes", "add_to_favorites",
     {}),

    ("Save to favorites", "add_to_favorites",
     {}),

    ("Add this to my collection", "add_to_favorites",
     {}),

    ("I want to save this", "add_to_favorites",
     {}),

    ("Mark as favorite", "add_to_favorites",
     {}),

    ("Add to favorite songs", "add_to_favorites",
     {}),

    ("Save this track", "add_to_favorites",
     {}),

    ("Like this track", "add_to_favorites",
     {}),

    ("Add to my favorites list", "add_to_favorites",
     {}),

    ("Save this one", "add_to_favorites",
     {}),

    ("I like this", "add_to_favorites",
     {}),

    ("Add this song to favorites", "add_to_favorites",
     {}),

    ("Shuffle this playlist", "shuffle_toggle",
     {'scope': 'playlist'}),

    ("Turn on shuffle", "shuffle_toggle",
     {}),

    ("Shuffle on", "shuffle_toggle",
     {}),

    ("Enable shuffle", "shuffle_toggle",
     {}),

    ("Shuffle the queue", "shuffle_toggle",
     {'scope': 'queue'}),

    ("Turn off shuffle", "shuffle_toggle",
     {}),

    ("Disable shuffle", "shuffle_toggle",
     {}),

    ("Shuffle off", "shuffle_toggle",
     {}),

    ("Shuffle this album", "shuffle_toggle",
     {'scope': 'album'}),

    ("Turn shuffle on", "shuffle_toggle",
     {}),

    ("Turn shuffle off", "shuffle_toggle",
     {}),

    ("Enable shuffle mode", "shuffle_toggle",
     {}),

    ("Disable shuffle mode", "shuffle_toggle",
     {}),

    ("Shuffle", "shuffle_toggle",
     {}),

    ("Toggle shuffle", "shuffle_toggle",
     {}),

    ("Shuffle playlist", "shuffle_toggle",
     {'scope': 'playlist'}),

    ("Shuffle album", "shuffle_toggle",
     {'scope': 'album'}),

    ("Shuffle queue", "shuffle_toggle",
     {'scope': 'queue'}),

    ("Mix up this playlist", "shuffle_toggle",
     {'scope': 'playlist'}),

    ("Randomize this playlist", "shuffle_toggle",
     {'scope': 'playlist'}),

    ("Put shuffle on", "shuffle_toggle",
     {}),

    ("Take shuffle off", "shuffle_toggle",
     {}),

    ("Start shuffling", "shuffle_toggle",
     {}),

    ("Stop shuffling", "shuffle_toggle",
     {}),

    ("Activate shuffle", "shuffle_toggle",
     {}),

    ("Repeat this song", "repeat_mode",
     {'mode': 'one'}),

    ("Loop this track", "repeat_mode",
     {'mode': 'one'}),

    ("Repeat all", "repeat_mode",
     {'mode': 'all'}),

    ("Turn on repeat", "repeat_mode",
     {'mode': 'all'}),

    ("Loop playlist", "repeat_mode",
     {'mode': 'playlist'}),

    ("Repeat one", "repeat_mode",
     {'mode': 'one'}),

    ("Turn off repeat", "repeat_mode",
     {'mode': 'off'}),

    ("Disable repeat", "repeat_mode",
     {'mode': 'off'}),

    ("Repeat off", "repeat_mode",
     {'mode': 'off'}),

    ("Enable repeat", "repeat_mode",
     {'mode': 'all'}),

    ("Loop this", "repeat_mode",
     {'mode': 'one'}),

    ("Repeat the playlist", "repeat_mode",
     {'mode': 'playlist'}),

    ("Repeat this one", "repeat_mode",
     {'mode': 'one'}),

    ("Repeat everything", "repeat_mode",
     {'mode': 'all'}),

    ("Keep repeating this song", "repeat_mode",
     {'mode': 'one'}),

    ("Loop all songs", "repeat_mode",
     {'mode': 'all'}),

    ("Turn repeat on", "repeat_mode",
     {'mode': 'all'}),

    ("Turn repeat off", "repeat_mode",
     {'mode': 'off'}),

    ("Start repeating", "repeat_mode",
     {'mode': 'all'}),

    ("Stop repeating", "repeat_mode",
     {'mode': 'off'}),

    ("Replay this song", "repeat_mode",
     {'mode': 'one'}),

    ("Replay all", "repeat_mode",
     {'mode': 'all'}),

    ("Repeat mode on", "repeat_mode",
     {'mode': 'all'}),

    ("Repeat mode off", "repeat_mode",
     {'mode': 'off'}),

    ("Loop song", "repeat_mode",
     {'mode': 'one'}),

    ("Play some jazz music", "play_genre",
     {'genre': 'jazz'}),

    ("Play rock", "play_genre",
     {'genre': 'rock'}),

    ("Play pop music", "play_genre",
     {'genre': 'pop'}),

    ("Play classical", "play_genre",
     {'genre': 'classical'}),

    ("Play hip hop", "play_genre",
     {'genre': 'hip hop'}),

    ("Play country music", "play_genre",
     {'genre': 'country'}),

    ("Play electronic music", "play_genre",
     {'genre': 'electronic'}),

    ("Play R&B", "play_genre",
     {'genre': 'R&B'}),

    ("Play indie music", "play_genre",
     {'genre': 'indie'}),

    ("Play metal", "play_genre",
     {'genre': 'metal'}),

    ("Play blues", "play_genre",
     {'genre': 'blues'}),

    ("Play reggae", "play_genre",
     {'genre': 'reggae'}),

    ("Play soul music", "play_genre",
     {'genre': 'soul'}),

    ("Play funk", "play_genre",
     {'genre': 'funk'}),

    ("Play disco", "play_genre",
     {'genre': 'disco'}),

    ("Play latin music", "play_genre",
     {'genre': 'latin'}),

    ("Play alternative", "play_genre",
     {'genre': 'alternative'}),

    ("Play punk", "play_genre",
     {'genre': 'punk'}),

    ("Play folk music", "play_genre",
     {'genre': 'folk'}),

    ("Play gospel", "play_genre",
     {'genre': 'gospel'}),

    ("Play some chill music", "play_genre",
     {'genre': 'chill', 'mood': 'chill'}),

    ("Play energetic music", "play_genre",
     {'mood': 'energetic'}),

    ("Play happy songs", "play_genre",
     {'mood': 'happy'}),

    ("Play sad music", "play_genre",
     {'mood': 'sad'}),

    ("Play relaxing music", "play_genre",
     {'mood': 'chill'}),

    ("Play workout music", "play_genre",
     {'genre': 'workout'}),

    ("Play party music", "play_genre",
     {'genre': 'party'}),

    ("Play romantic music", "play_genre",
     {'mood': 'romantic'}),

    ("Play upbeat music", "play_genre",
     {'mood': 'energetic'}),

    ("Play mellow music", "play_genre",
     {'mood': 'chill'}),

    ("Set sleep timer for 30 minutes", "set_sleep_timer",
     {'action': 'set', 'duration_minutes': 30}),

    ("Stop music in 1 hour", "set_sleep_timer",
     {'action': 'set', 'duration_minutes': 60}),

    ("Sleep timer 45 minutes", "set_sleep_timer",
     {'action': 'set', 'duration_minutes': 45}),

    ("Turn off music in 20 minutes", "set_sleep_timer",
     {'action': 'set', 'duration_minutes': 20}),

    ("Set timer for 15 minutes", "set_sleep_timer",
     {'action': 'set', 'duration_minutes': 15}),

    ("Cancel sleep timer", "set_sleep_timer",
     {'action': 'cancel'}),

    ("Remove sleep timer", "set_sleep_timer",
     {'action': 'cancel'}),

    ("Check sleep timer", "set_sleep_timer",
     {'action': 'check'}),

    ("How much time left on timer", "set_sleep_timer",
     {'action': 'check'}),

    ("Set sleep timer 10 minutes", "set_sleep_timer",
     {'action': 'set', 'duration_minutes': 10}),

    ("Stop playing in 25 minutes", "set_sleep_timer",
     {'action': 'set', 'duration_minutes': 25}),

    ("Set timer for 40 minutes", "set_sleep_timer",
     {'action': 'set', 'duration_minutes': 40}),

    ("Turn off in 35 minutes", "set_sleep_timer",
     {'action': 'set', 'duration_minutes': 35}),

    ("Sleep timer 50 minutes", "set_sleep_timer",
     {'action': 'set', 'duration_minutes': 50}),

    ("Set sleep timer 90 minutes", "set_sleep_timer",
     {'action': 'set', 'duration_minutes': 90}),

    ("Cancel timer", "set_sleep_timer",
     {'action': 'cancel'}),

    ("Stop timer", "set_sleep_timer",
     {'action': 'cancel'}),

    ("Check timer status", "set_sleep_timer",
     {'action': 'check'}),

    ("Show sleep timer", "set_sleep_timer",
     {'action': 'check'}),

    ("Set timer 5 minutes", "set_sleep_timer",
     {'action': 'set', 'duration_minutes': 5}),

    ("Stop music in 2 hours", "set_sleep_timer",
     {'action': 'set', 'duration_minutes': 120}),

    ("Turn off after 55 minutes", "set_sleep_timer",
     {'action': 'set', 'duration_minutes': 55}),

    ("Delete sleep timer", "set_sleep_timer",
     {'action': 'cancel'}),

    ("What's the sleep timer", "set_sleep_timer",
     {'action': 'check'}),

    ("Set sleep timer 75 minutes", "set_sleep_timer",
     {'action': 'set', 'duration_minutes': 75}),

    ("Play radio like Coldplay", "play_artist_radio",
     {'artist': 'Coldplay'}),

    ("Play Taylor Swift radio", "play_artist_radio",
     {'artist': 'Taylor Swift'}),

    ("Radio based on Drake", "play_artist_radio",
     {'artist': 'Drake'}),

    ("Play station like The Weeknd", "play_artist_radio",
     {'artist': 'The Weeknd'}),

    ("Play Ariana Grande radio", "play_artist_radio",
     {'artist': 'Ariana Grande'}),

    ("Start Ed Sheeran radio", "play_artist_radio",
     {'artist': 'Ed Sheeran'}),

    ("Play Billie Eilish radio", "play_artist_radio",
     {'artist': 'Billie Eilish'}),

    ("Radio like Post Malone", "play_artist_radio",
     {'artist': 'Post Malone'}),

    ("Play Bruno Mars station", "play_artist_radio",
     {'artist': 'Bruno Mars'}),

    ("Start Dua Lipa radio", "play_artist_radio",
     {'artist': 'Dua Lipa'}),

    ("Play radio based on Adele", "play_artist_radio",
     {'artist': 'Adele'}),

    ("Start radio like Imagine Dragons", "play_artist_radio",
     {'artist': 'Imagine Dragons'}),

    ("Play Justin Bieber radio", "play_artist_radio",
     {'artist': 'Justin Bieber'}),

    ("Radio similar to Queen", "play_artist_radio",
     {'artist': 'Queen'}),

    ("Play Beatles radio", "play_artist_radio",
     {'artist': 'Beatles'}),

    ("Start station like Michael Jackson", "play_artist_radio",
     {'artist': 'Michael Jackson'}),

    ("Play radio like Eagles", "play_artist_radio",
     {'artist': 'Eagles'}),

    ("Start Fleetwood Mac radio", "play_artist_radio",
     {'artist': 'Fleetwood Mac'}),

    ("Play station like Led Zeppelin", "play_artist_radio",
     {'artist': 'Led Zeppelin'}),

    ("Radio based on Pink Floyd", "play_artist_radio",
     {'artist': 'Pink Floyd'}),

    ("Play Nirvana radio", "play_artist_radio",
     {'artist': 'Nirvana'}),

    ("Start AC/DC station", "play_artist_radio",
     {'artist': 'AC/DC'}),

    ("Play radio like Metallica", "play_artist_radio",
     {'artist': 'Metallica'}),

    ("Start radio based on Guns N Roses", "play_artist_radio",
     {'artist': 'Guns N Roses'}),

    ("Play station similar to Oasis", "play_artist_radio",
     {'artist': 'Oasis'}),

    ("Download this song", "download_for_offline",
     {'content_type': 'song', 'name': 'this song'}),

    ("Download this album", "download_for_offline",
     {'content_type': 'album', 'name': 'this album'}),

    ("Download Workout Mix playlist", "download_for_offline",
     {'content_type': 'playlist', 'name': 'Workout Mix'}),

    ("Save this song offline", "download_for_offline",
     {'content_type': 'song', 'name': 'this song'}),

    ("Download for offline", "download_for_offline",
     {'content_type': 'song', 'name': 'current'}),

    ("Make this available offline", "download_for_offline",
     {'content_type': 'song', 'name': 'this song'}),

    ("Download this track", "download_for_offline",
     {'content_type': 'song', 'name': 'this track'}),

    ("Save album offline", "download_for_offline",
     {'content_type': 'album', 'name': 'this album'}),

    ("Download playlist", "download_for_offline",
     {'content_type': 'playlist', 'name': 'current playlist'}),

    ("Make available offline", "download_for_offline",
     {'content_type': 'song', 'name': 'this'}),

    ("Download this in high quality", "download_for_offline",
     {'content_type': 'song', 'name': 'this song', 'quality': 'high'}),

    ("Save this song in medium quality", "download_for_offline",
     {'content_type': 'song', 'name': 'this song', 'quality': 'medium'}),

    ("Download album high quality", "download_for_offline",
     {'content_type': 'album', 'name': 'this album', 'quality': 'high'}),

    ("Save playlist for offline", "download_for_offline",
     {'content_type': 'playlist', 'name': 'current playlist'}),

    ("Download song", "download_for_offline",
     {'content_type': 'song', 'name': 'current song'}),

    ("Save this offline", "download_for_offline",
     {'content_type': 'song', 'name': 'this'}),

    ("Download to device", "download_for_offline",
     {'content_type': 'song', 'name': 'current'}),

    ("Make this downloadable", "download_for_offline",
     {'content_type': 'song', 'name': 'this'}),

    ("Save for offline listening", "download_for_offline",
     {'content_type': 'song', 'name': 'this'}),

    ("Download current song", "download_for_offline",
     {'content_type': 'song', 'name': 'current song'}),

    ("Save current album", "download_for_offline",
     {'content_type': 'album', 'name': 'current album'}),

    ("Download whole playlist", "download_for_offline",
     {'content_type': 'playlist', 'name': 'this playlist'}),

    ("Make playlist available offline", "download_for_offline",
     {'content_type': 'playlist', 'name': 'this playlist'}),

    ("Download this album offline", "download_for_offline",
     {'content_type': 'album', 'name': 'this album'}),

    ("Save track offline", "download_for_offline",
     {'content_type': 'song', 'name': 'this track'}),

    ("Show lyrics", "show_lyrics",
     {}),

    ("Display lyrics", "show_lyrics",
     {}),

    ("What are the lyrics", "show_lyrics",
     {}),

    ("Show me the lyrics", "show_lyrics",
     {}),

    ("Lyrics", "show_lyrics",
     {}),

    ("Show song lyrics", "show_lyrics",
     {}),

    ("Display song lyrics", "show_lyrics",
     {}),

    ("What are the words", "show_lyrics",
     {}),

    ("Show me the words", "show_lyrics",
     {}),

    ("Get lyrics", "show_lyrics",
     {}),

    ("Open lyrics", "show_lyrics",
     {}),

    ("View lyrics", "show_lyrics",
     {}),

    ("Show lyrics for this song", "show_lyrics",
     {}),

    ("Display lyrics for current song", "show_lyrics",
     {}),

    ("Show me lyrics for this", "show_lyrics",
     {}),

    ("What's the lyrics", "show_lyrics",
     {}),

    ("Can I see the lyrics", "show_lyrics",
     {}),

    ("I want to see lyrics", "show_lyrics",
     {}),

    ("Show text", "show_lyrics",
     {}),

    ("Display text", "show_lyrics",
     {}),

    ("Lyrics please", "show_lyrics",
     {}),

    ("Show words", "show_lyrics",
     {}),

    ("Get song lyrics", "show_lyrics",
     {}),

    ("Open song lyrics", "show_lyrics",
     {}),

    ("View song lyrics", "show_lyrics",
     {}),

    ("Share this on Instagram", "share_song",
     {'platform': 'instagram'}),

    ("Share on Twitter", "share_song",
     {'platform': 'twitter'}),

    ("Share to Facebook", "share_song",
     {'platform': 'facebook'}),

    ("Send this song", "share_song",
     {'platform': 'message'}),

    ("Copy link", "share_song",
     {'platform': 'copy_link'}),

    ("Share this song on Instagram", "share_song",
     {'platform': 'instagram'}),

    ("Post to Twitter", "share_song",
     {'platform': 'twitter'}),

    ("Share on FB", "share_song",
     {'platform': 'facebook'}),

    ("Message this", "share_song",
     {'platform': 'message'}),

    ("Copy song link", "share_song",
     {'platform': 'copy_link'}),

    ("Share on insta", "share_song",
     {'platform': 'instagram'}),

    ("Tweet this", "share_song",
     {'platform': 'twitter'}),

    ("Post on Facebook", "share_song",
     {'platform': 'facebook'}),

    ("Send via message", "share_song",
     {'platform': 'message'}),

    ("Get link", "share_song",
     {'platform': 'copy_link'}),

    ("Share song on Instagram", "share_song",
     {'platform': 'instagram'}),

    ("Share this on Twitter", "share_song",
     {'platform': 'twitter'}),

    ("Post this on Facebook", "share_song",
     {'platform': 'facebook'}),

    ("Text this song", "share_song",
     {'platform': 'message'}),

    ("Copy this link", "share_song",
     {'platform': 'copy_link'}),

    ("Share track on Instagram", "share_song",
     {'platform': 'instagram'}),

    ("Share track on Twitter", "share_song",
     {'platform': 'twitter'}),

    ("Post track on Facebook", "share_song",
     {'platform': 'facebook'}),

    ("Share via message", "share_song",
     {'platform': 'message'}),

    ("Copy track link", "share_song",
     {'platform': 'copy_link'}),

    ("Boost the bass", "adjust_equalizer",
     {'preset': 'bass_boost'}),

    ("Vocal boost mode", "adjust_equalizer",
     {'preset': 'vocal_boost'}),

    ("Set EQ to flat", "adjust_equalizer",
     {'preset': 'flat'}),

    ("Rock EQ", "adjust_equalizer",
     {'preset': 'rock'}),

    ("Pop equalizer", "adjust_equalizer",
     {'preset': 'pop'}),

    ("Jazz EQ", "adjust_equalizer",
     {'preset': 'jazz'}),

    ("Classical equalizer", "adjust_equalizer",
     {'preset': 'classical'}),

    ("Bass boost", "adjust_equalizer",
     {'preset': 'bass_boost'}),

    ("Enhance vocals", "adjust_equalizer",
     {'preset': 'vocal_boost'}),

    ("Flat equalizer", "adjust_equalizer",
     {'preset': 'flat'}),

    ("Set EQ to rock", "adjust_equalizer",
     {'preset': 'rock'}),

    ("Switch to pop EQ", "adjust_equalizer",
     {'preset': 'pop'}),

    ("Change to jazz preset", "adjust_equalizer",
     {'preset': 'jazz'}),

    ("Classical preset", "adjust_equalizer",
     {'preset': 'classical'}),

    ("More bass", "adjust_equalizer",
     {'preset': 'bass_boost'}),

    ("Better vocals", "adjust_equalizer",
     {'preset': 'vocal_boost'}),

    ("Neutral EQ", "adjust_equalizer",
     {'preset': 'flat'}),

    ("Rock preset", "adjust_equalizer",
     {'preset': 'rock'}),

    ("Pop preset", "adjust_equalizer",
     {'preset': 'pop'}),

    ("Jazz preset", "adjust_equalizer",
     {'preset': 'jazz'}),

    ("Classical mode", "adjust_equalizer",
     {'preset': 'classical'}),

    ("Increase bass", "adjust_equalizer",
     {'preset': 'bass_boost'}),

    ("Increase vocals", "adjust_equalizer",
     {'preset': 'vocal_boost'}),

    ("Default EQ", "adjust_equalizer",
     {'preset': 'flat'}),

    ("Set equalizer to rock", "adjust_equalizer",
     {'preset': 'rock'}),

    ("Play this next", "queue_song",
     {'song_name': 'this song', 'position': 'next'}),

    ("Add to queue", "queue_song",
     {'song_name': 'this song'}),

    ("Queue this", "queue_song",
     {'song_name': 'this song'}),

    ("Add this to queue", "queue_song",
     {'song_name': 'this song'}),

    ("Play next", "queue_song",
     {'song_name': 'this song', 'position': 'next'}),

    ("Add to end of queue", "queue_song",
     {'song_name': 'this song', 'position': 'end'}),

    ("Queue this song", "queue_song",
     {'song_name': 'this song'}),

    ("Add this next", "queue_song",
     {'song_name': 'this song', 'position': 'next'}),

    ("Play after this", "queue_song",
     {'song_name': 'this song', 'position': 'next'}),

    ("Add to playback queue", "queue_song",
     {'song_name': 'this song'}),

    ("Queue up this", "queue_song",
     {'song_name': 'this song'}),

    ("Play this after current", "queue_song",
     {'song_name': 'this song', 'position': 'next'}),

    ("Add this track to queue", "queue_song",
     {'song_name': 'this track'}),

    ("Queue track", "queue_song",
     {'song_name': 'this track'}),

    ("Add next in queue", "queue_song",
     {'song_name': 'this song', 'position': 'next'}),

    ("Put in queue", "queue_song",
     {'song_name': 'this song'}),

    ("Add to play queue", "queue_song",
     {'song_name': 'this song'}),

    ("Queue song", "queue_song",
     {'song_name': 'this song'}),

    ("Play this second", "queue_song",
     {'song_name': 'this song', 'position': 'next'}),

    ("Add at end", "queue_song",
     {'song_name': 'this song', 'position': 'end'}),

    ("Queue at end", "queue_song",
     {'song_name': 'this song', 'position': 'end'}),

    ("Play after", "queue_song",
     {'song_name': 'this song', 'position': 'next'}),

    ("Add this up next", "queue_song",
     {'song_name': 'this song', 'position': 'next'}),

    ("Queue for later", "queue_song",
     {'song_name': 'this song', 'position': 'end'}),

    ("Add to waiting list", "queue_song",
     {'song_name': 'this song'}),

    ("Who sings this", "get_song_info",
     {'info_type': 'artist'}),

    ("What's this song called", "get_song_info",
     {'info_type': 'all'}),

    ("Song info", "get_song_info",
     {'info_type': 'all'}),

    ("Who is the artist", "get_song_info",
     {'info_type': 'artist'}),

    ("What album is this", "get_song_info",
     {'info_type': 'album'}),

    ("When was this released", "get_song_info",
     {'info_type': 'year'}),

    ("What genre is this", "get_song_info",
     {'info_type': 'genre'}),

    ("How long is this song", "get_song_info",
     {'info_type': 'duration'}),

    ("Song details", "get_song_info",
     {'info_type': 'all'}),

    ("Tell me about this song", "get_song_info",
     {'info_type': 'all'}),

    ("What's the song name", "get_song_info",
     {}),

    ("Who's singing", "get_song_info",
     {'info_type': 'artist'}),

    ("Album name", "get_song_info",
     {'info_type': 'album'}),

    ("What year", "get_song_info",
     {'info_type': 'year'}),

    ("Song genre", "get_song_info",
     {'info_type': 'genre'}),

    ("Track info", "get_song_info",
     {'info_type': 'all'}),

    ("Show song info", "get_song_info",
     {'info_type': 'all'}),

    ("What's playing", "get_song_info",
     {'info_type': 'all'}),

    ("Current song info", "get_song_info",
     {'info_type': 'all'}),

    ("Who made this", "get_song_info",
     {'info_type': 'artist'}),

    ("What's the album", "get_song_info",
     {'info_type': 'album'}),

    ("Release year", "get_song_info",
     {'info_type': 'year'}),

    ("Genre", "get_song_info",
     {'info_type': 'genre'}),

    ("Song length", "get_song_info",
     {'info_type': 'duration'}),

    ("Track details", "get_song_info",
     {'info_type': 'all'}),

    ("Show me song details", "get_song_info",
     {'info_type': 'all'}),

    ("What song is this", "get_song_info",
     {}),

    ("Who's the artist", "get_song_info",
     {'info_type': 'artist'}),

    ("Which album", "get_song_info",
     {'info_type': 'album'}),

    ("Song year", "get_song_info",
     {'info_type': 'year'}),

]
