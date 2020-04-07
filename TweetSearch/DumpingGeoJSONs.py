import json

region_to_dump = input("What region are you dumping data for?\n").lower()
print("Please paste your json file to be dumped:\n")
content = []
keepIterating = True
while keepIterating:
    line = input()
    if line:
        content.append(line)
    else:
    	keepIterating = False
    	break

json_to_dump = '\n'.join(content).replace('\n','').replace(' ','')

dic = json.loads(json_to_dump)

print("JSON correctly loaded.")

SW_long = dic["features"][0]['geometry']['coordinates'][0][0][0]
SW_lat = dic["features"][0]['geometry']['coordinates'][0][0][1]
NE_long = dic["features"][0]['geometry']['coordinates'][0][2][0]
NE_lat = dic["features"][0]['geometry']['coordinates'][0][2][1]

with open("RegionCoords.json", 'a') as file: # a mode stands for appending so that file is not overwritten
    file.write('{"Region": "'+region_to_dump+
               '","SWlongitude":'+str(SW_long)+
               ',"SWlatitude":'+str(SW_lat)+
               ',"NElongitude":'+str(NE_long)+
               ',"NElatitude":'+str(NE_lat)+'}\n')

print("Your new entry has been added to RegionCoords.json")