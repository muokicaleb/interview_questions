# Practical Interview Questions

Practical interview questions curated from the internet.<br>
*Disclaimer* I'm not in HR or any hiring team. <br>
Sources
* [https://www.dailycodingproblem.com/](https://www.dailycodingproblem.com/)


1. Given a string, return the first recurring character in it, or null if there is no recurring character. For example, given the string "acbbac", return "b". Given the string "abcdef", return null.

```
def find_recurring(text):
    """ A function to find first recurring character in
    a string.
    """
    string_list = list(text)
    temp_list = []
    for item in string_list:
        if item in temp_list:
            return item
        elif item not in temp_list:
            temp_list.append(item)
    return 'null'     

```