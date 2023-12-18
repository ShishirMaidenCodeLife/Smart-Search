with open("requirements.txt", "r") as file:
    reqq=[line.split("@")[0] for line in file]
 

with open("new_req.txt", "w") as file:
    for i in reqq:
        file.write(i)