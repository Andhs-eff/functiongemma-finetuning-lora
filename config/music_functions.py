# Music Application Function Definitions

MUSIC_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "play_song",
            "description": "Play a specific song by name or artist",
            "parameters": {
                "type": "object",
                "properties": {
                    "song_name": {
                        "type": "string",
                        "description": "Name of the song to play"
                    },
                    "artist": {
                        "type": "string",
                        "description": "Artist name (optional)"
                    },
                    "album": {
                        "type": "string",
                        "description": "Album name (optional)"
                    }
                },
                "required": ["song_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "playback_control",
            "description": "Control music playback (play, pause, skip, previous, stop)",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["play", "pause", "skip", "next", "previous", "stop", "resume"],
                        "description": "Playback action to perform"
                    }
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_playlist",
            "description": "Create a new playlist",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name for the new playlist"
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional description of the playlist"
                    },
                    "songs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Initial songs to add (optional)"
                    }
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_playlist",
            "description": "Add song(s) to an existing playlist",
            "parameters": {
                "type": "object",
                "properties": {
                    "playlist_name": {
                        "type": "string",
                        "description": "Name of the playlist"
                    },
                    "song_name": {
                        "type": "string",
                        "description": "Song to add"
                    },
                    "artist": {
                        "type": "string",
                        "description": "Artist of the song (optional)"
                    }
                },
                "required": ["playlist_name", "song_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_music",
            "description": "Search for songs, artists, albums, or playlists",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "type": {
                        "type": "string",
                        "enum": ["song", "artist", "album", "playlist", "all"],
                        "description": "Type of content to search for"
                    },
                    "genre": {
                        "type": "string",
                        "description": "Filter by genre (optional)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_favorites",
            "description": "Add current or specified song to favorites/liked songs",
            "parameters": {
                "type": "object",
                "properties": {
                    "song_name": {
                        "type": "string",
                        "description": "Song name (uses current song if not specified)"
                    },
                    "artist": {
                        "type": "string",
                        "description": "Artist name (optional)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_volume",
            "description": "Adjust volume level",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "integer",
                        "description": "Volume level (0-100) or relative (+/-10)",
                        "minimum": 0,
                        "maximum": 100
                    },
                    "action": {
                        "type": "string",
                        "enum": ["set", "increase", "decrease", "mute", "unmute"],
                        "description": "Volume action"
                    }
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "shuffle_toggle",
            "description": "Toggle shuffle mode on or off",
            "parameters": {
                "type": "object",
                "properties": {
                    "state": {
                        "type": "boolean",
                        "description": "True for on, False for off, null to toggle"
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["playlist", "album", "queue"],
                        "description": "What to shuffle"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "repeat_mode",
            "description": "Set repeat/loop mode",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["off", "one", "all", "playlist"],
                        "description": "Repeat mode: off, repeat one song, or repeat all"
                    }
                },
                "required": ["mode"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "play_genre",
            "description": "Play music from a specific genre",
            "parameters": {
                "type": "object",
                "properties": {
                    "genre": {
                        "type": "string",
                        "description": "Music genre (rock, pop, jazz, classical, etc.)"
                    },
                    "mood": {
                        "type": "string",
                        "description": "Optional mood (energetic, chill, sad, happy)"
                    }
                },
                "required": ["genre"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_sleep_timer",
            "description": "Set timer to stop music after specified duration",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_minutes": {
                        "type": "integer",
                        "description": "Minutes until music stops"
                    },
                    "action": {
                        "type": "string",
                        "enum": ["set", "cancel", "check"],
                        "description": "Timer action"
                    }
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "play_artist_radio",
            "description": "Play radio station based on artist or song",
            "parameters": {
                "type": "object",
                "properties": {
                    "artist": {
                        "type": "string",
                        "description": "Artist name to base radio on"
                    },
                    "song": {
                        "type": "string",
                        "description": "Song name to base radio on (alternative)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "download_for_offline",
            "description": "Download song, album, or playlist for offline listening",
            "parameters": {
                "type": "object",
                "properties": {
                    "content_type": {
                        "type": "string",
                        "enum": ["song", "album", "playlist"],
                        "description": "Type of content to download"
                    },
                    "name": {
                        "type": "string",
                        "description": "Name of song/album/playlist"
                    },
                    "quality": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Download quality"
                    }
                },
                "required": ["content_type", "name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "show_lyrics",
            "description": "Display lyrics for current or specified song",
            "parameters": {
                "type": "object",
                "properties": {
                    "song_name": {
                        "type": "string",
                        "description": "Song name (uses current song if not specified)"
                    },
                    "artist": {
                        "type": "string",
                        "description": "Artist name (optional)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "share_song",
            "description": "Share current or specified song",
            "parameters": {
                "type": "object",
                "properties": {
                    "song_name": {
                        "type": "string",
                        "description": "Song to share"
                    },
                    "platform": {
                        "type": "string",
                        "enum": ["instagram", "twitter", "facebook", "message", "copy_link"],
                        "description": "Where to share"
                    }
                },
                "required": ["platform"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_equalizer",
            "description": "Adjust audio equalizer settings",
            "parameters": {
                "type": "object",
                "properties": {
                    "preset": {
                        "type": "string",
                        "enum": ["flat", "rock", "pop", "jazz", "classical", "bass_boost", "vocal_boost"],
                        "description": "Equalizer preset"
                    },
                    "custom": {
                        "type": "object",
                        "description": "Custom EQ settings (optional)"
                    }
                },
                "required": ["preset"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "queue_song",
            "description": "Add song to play queue",
            "parameters": {
                "type": "object",
                "properties": {
                    "song_name": {
                        "type": "string",
                        "description": "Song to add to queue"
                    },
                    "artist": {
                        "type": "string",
                        "description": "Artist name (optional)"
                    },
                    "position": {
                        "type": "string",
                        "enum": ["next", "end"],
                        "description": "Where to add in queue"
                    }
                },
                "required": ["song_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_song_info",
            "description": "Get information about current or specified song",
            "parameters": {
                "type": "object",
                "properties": {
                    "song_name": {
                        "type": "string",
                        "description": "Song name (uses current song if not specified)"
                    },
                    "info_type": {
                        "type": "string",
                        "enum": ["artist", "album", "year", "genre", "duration", "all"],
                        "description": "Type of information to retrieve"
                    }
                },
                "required": []
            }
        }
    }
]

# System message for the music assistant
SYSTEM_MESSAGE = "You are a music application assistant that can control playback, manage playlists, search music, and help users enjoy their music experience."
