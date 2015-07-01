#!/usr/bin/env python3

from pprint import pprint
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from time import time, sleep
import requests
import sys
import re
import os.path
import json
from functools import reduce
import operator
import dateutil.parser
import datetime

def cache_result(fn, f, *args, **kwargs):
    if os.path.exists(fn):
        with open(fn, 'r') as f:
            return json.load(f)
    else:
        dn = os.path.dirname(fn)
        if dn: os.makedirs(dn, exist_ok=True)
        x = f(*args, **kwargs)
        with open(fn, 'w') as f:
            json.dump(x, f)
        return x

def matcher(template, s):
    def frac(s):
        try:
            a, b = s.split('/')
            return float(a) / float(b)
        except ValueError:
            return float(s)
    result = []
    types = tuple(x[1] for x in re.findall('%[dfz]', template))
    repl = (('(', r'\('), (')', r'\)'), ('*', '.*'), ('?', '.'), ('$', r'\$'), ('%d', '(\d+)'), ('%f', '(\d+|\d+\.\d+)'), ('%z', '(\d+|\d+/\d+)'))
    regexp = template
    for a, b in repl: regexp = regexp.replace(a, b)
    try:
        m = re.match(regexp, s)
    except Exception as e:
        print('matcher: bad regexp: {}'.format(regexp))
        raise e
    if m:
        for i, t in enumerate(types):
            z = m.group(i + 1)
            if t == 'd': z = int(z)
            elif t == 'f': z = float(z)
            elif t == 'z': z = frac(z)
            result.append(z)
    if not result:
        print('matcher: unable to match "{}" (compiled: "{}") to string "{}"'.format(template, regexp, s))
    return result

def date2float(dt):
    y, m, d = map(int, dt.split('-'))
    return y + (datetime.datetime(y,m,d)-datetime.datetime(y,1,1)).days/367.0

class DPReviewCrawler:
    def __init__(self):
        self.userAgent = 'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.120 Safari/537.36'

        self.fetchDiscontinuedProducts = True

        self.minInterval = 2 # seconds
        self.lastCrawlTime = 0

        self.brand_links = cache_result('brand_links.json', self.fetch_brand_links)
        self.product_links = cache_result('product_links.json', self.fetch_product_links)

    def log(self, s):
        sys.stdout.write(s)
        sys.stdout.flush()

    def get_url(self, url, local_file):
        self.log('*** get %s... ' % urlparse(url).path)
        if os.path.exists(local_file):
            self.log('cached\n')
            with open(local_file, 'rb') as f:
                return f.read()
        else:
            t = time()
            d = self.minInterval - (t - self.lastCrawlTime)
            if d > 0:
                self.log('wait {:.1f}s... '.format(d))
                sleep(d)
            self.lastCrawlTime = t
            self.log('fetch... ')
            response = requests.get(url, headers={'User-agent': self.userAgent})
            dn = os.path.dirname(local_file)
            if dn: os.makedirs(dn, exist_ok=True)
            with open(local_file, 'wb') as f:
                f.write(response.content)
            self.log('ok\n')
            return response.content

    def get_soup(self, url, local_file):
        return BeautifulSoup(self.get_url(url, local_file))

    def url_key(self, url):
        return os.path.split(urlparse(url).path)[1]

    def fetch_brand_links(self):
        brand_index = 'http://www.dpreview.com/products/'
        soup = self.get_soup(brand_index, 'htm/brand_index.htm')
        brand_links = {}
        for a in soup.select('div#mainContent div.brands a.brand'):
            url = urljoin(brand_index, a['href'])
            key = self.url_key(url)
            brand_links[key] = url
        return brand_links

    def fetch_product_links(self):
        product_links = {}
        for brand_key in self.brand_links:
            brand_link = self.brand_links[brand_key]
            soup = self.get_soup(brand_link, 'htm/%s/main.htm' % brand_key)
            cc = soup.select('div#mainContent div.categories div.categoryCameras div.link a')
            if not cc: continue
            soup = self.get_soup(cc[0]['href'], 'htm/%s/cameras.htm' % brand_key)
            product_links[brand_key] = {}
            for a in soup.select('div#mainContent div.activeProducts div.info div.name a'):
                url = urljoin(brand_link, a['href'])
                key = self.url_key(url)
                product_links[brand_key][key] = url
            if self.fetchDiscontinuedProducts:
                for a in soup.select('div#mainContent div.discontinuedProducts li.product span.name a'):
                    url = urljoin(brand_link, a['href'])
                    key = self.url_key(url)
                    product_links[brand_key][key] = url
        return product_links

    def fetch_product_specs(self, brand_key, product_key):
        specs = {}

        product_link = self.product_links[brand_key][product_key]
        soup = self.get_soup(product_link + '/specifications', 'htm/%s/%s/specs.htm' % (brand_key, product_key))
        
        for tr in soup.select('div#mainContent div.specificationsPage table.specification tbody tr'):
            k = tr.select('th.label')[0].text.strip()
            ev = tr.select('td.value')[0]
            if ev.find('li'):
                v = []
                for ev1 in ev.select('li'):
                    v.append(ev1.text.strip().replace(';', ','))
                v = '; '.join(v)
            else:
                v = ev.text.strip()
            specs[k] = v
        if not specs:
            print('warning: found no specs for %s' % product_key)
        specs['Name'] = soup.select('div#mainContent div.headerContainer h1')[0].text.strip()
        ad = list(soup.select('div#mainContent div.headerContainer div.shortSpecs')[0].stripped_strings)[1]
        try:
            ad = ad[:ad.index('\u2022')]
        except:
            pass
        ad = ad.strip()
        try:
            specs['Announce date'] = dateutil.parser.parse(ad).date().isoformat()
        except Exception as e:
            print('warning: failed to parse date "{}"'.format(ad))
            #raise e

        soup = self.get_soup(product_link, 'htm/%s/%s/main.htm' % (brand_key, product_key))
        price = soup.select('div#mainContent div.rightColumn div.smallBuybox div.price')
        if price:
            price = price[0]
            #print(price); print('class: %s' % price[0]['class'])
            if 'range' in price['class']:
                price_start = price.select('span.start')[0].text.strip()
                price_end = price.select('span.end')[0].text.strip()
            else:
                price_start, price_end = 2 * [price.text.strip()]
            specs['Price Min'] = price_start.replace(',','')
            specs['Price Max'] = price_end.replace(',','')

        return specs

    def get_product_specs(self, brand_key, product_key):
        return cache_result('specs/%s/%s.json' % (brand_key, product_key), self.fetch_product_specs, brand_key, product_key)

    def get_all_product_specs(self):
        self.specs = {}
        for bk in c.product_links:
            for pk in c.product_links[bk]:
                self.specs[pk] = self.get_product_specs(bk, pk)

    def get_specs_statistics(self):
        from collections import Counter
        counter = Counter()
        for bk in c.product_links:
            for pk in c.product_links[bk]:
                counter.update(self.specs[pk].keys())
        return sorted(((v, k) for k, v in counter.items()), key=lambda x: -x[0])

    def project_specs(self, key):
        for bk in c.product_links:
            for pk in c.product_links[bk]:
                try:
                    yield self.specs[pk][key]
                except KeyError:
                    pass

    def vectorize_specs(self, k, s):
        v = [k] + \
                matcher('%d megapixels', s['Effective pixels']) + \
                [reduce(operator.mul, matcher('*(%f x %f mm)', s['Sensor size']))] + \
                matcher('%z sec', s['Maximum shutter speed']) + \
                matcher('%d g *', s['Weight (inc. batteries)']) + \
                matcher('$%f', s['Price Min']) + \
                matcher('$%f', s['Price Max']) + \
                [date2float(s['Announce date'])]

        # safety check:
        list(map(float, v[1:]))
        return v


if __name__ == '__main__':
    c = DPReviewCrawler()
    c.get_all_product_specs()

    cmd = sys.argv[1] if len(sys.argv) >= 2 else None

    if cmd == 'project':
        for i in c.project_specs(sys.argv[2]):
            print(i)
    elif cmd == 'stats':
        stats = c.get_specs_statistics()
        for i in stats:
            print(*i)
    elif cmd == 'dataset':
        print(','.join(['name','megapixels','sensor_area','max_shutter_spd','weight','price_min','price_max','year']))
        for pk, specs in c.specs.items():
            try:
                print(','.join(str(x) for x in c.vectorize_specs(pk, specs)))
            except KeyError:
                pass
    else:
        print('''usage:

    dpreview.py project <field>
        prints field <field> of every record

    dpreview.py stats
        prints statistics of field usage

    dpreview.py dataset
        creates csv dataset
''')
