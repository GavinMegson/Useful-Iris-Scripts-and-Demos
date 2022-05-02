    for ch in [0, 1]:
        for i in range(M):
            schedule = "GG"*(i+ch) + "PG" + "GG"*(M*2-(i+ch))
            conf = {"tdd_enabled" : True,
                    "frame_mode" : "free_running",
                    "symbol_size" : symSamp,
                    "frames" : schedule,
