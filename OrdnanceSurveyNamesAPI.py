import urllib.request as urllib
import urllib.error as urlliberr
import urllib.parse as urllibenc
import collections

"""
PERSONAL ADAPTATION FOR NON-COMMERCIAL PURPOSES
FROM THE ORDNANCE SURVEY API GITHUB PAGE
https://github.com/OrdnanceSurvey/OS-Open-Names
UNDER THE Apache LICENCE
"""


def ordnance_survey_location_query(query):

    # DEBUG
    # print('CONNECTION TO ORDNANCE SURVEY DATABASE')
    values = {'key': 'XySppRTaF9Nsb2XASeTU0tw9Tc3GyZvB', 'query': query}
    url = 'https://api.ordnancesurvey.co.uk/opennames/v1/find'
    data = urllibenc.urlencode(values)
    full_url = url + '?' + data

    # DEBUG
    # print(full_url)

    try:
        f = urllib.urlopen(full_url)
    except urlliberr.HTTPError as e:
        if e.code == 401:
            print('401 not authorized')
        elif e.code == 404:
            print('404 not found')
        elif e.code == 503:
            print('service unavailable')
        else:
            print('unknown error: ')
    else:
        # DEBUG
        # print('success')

        response = f.read()
        response_count = 0
        for line in response.splitlines():
            word_lst = line.decode().split(':')
            for word in word_lst:

                if response_count < 10:

                    if 'ID' and '"NAME1" ' and "populatedPlace" in word:
                        response_count += 1
                        # DEBUG
                        # print('-' * 80)
                        # print(line)
        f.close()
        # DEBUG
        # print(response_count)
        return response_count

