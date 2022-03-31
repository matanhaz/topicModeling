import xml.etree.ElementTree as ET
import os

tree = ET.parse('SearchRequest.xml')
root = tree.getroot()
titles = []
for i in root[0].findall('item'):
    titles.append(i.find('key').text)

if not os.path.exists('bugs'):
    os.mkdir('bugs')

for item in titles:
    tree = ET.parse('SearchRequest.xml')
    root = tree.getroot()
    for i in root[0].findall('item'):
        if i.find('key').text != item:
            root[0].remove(i)


    tree.write(f'bugs\\{item}.xml')

#
# for i in root[0].findall('item'):
#     root[0].remove(i)
#
# context = ET.iterparse('SearchRequest.xml', events=('start', ))
# b = context.findall("item")
#
# text = ""
# for event, elem in context:
#     print(event, elem)
#     if elem.tag != 'item':
#         pass
#
#     else:
#         title = elem.find('key').text
#         filename = format(title + ".xml")
#         with open(filename, 'wb') as f:
#             f.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
#             f.write(ET.tostring(elem))
