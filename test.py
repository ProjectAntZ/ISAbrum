import re

msg = "Sensors: 1; Distance: 234"
match = re.search('Sensors: (\d+)', msg).group(1)
print(match)
