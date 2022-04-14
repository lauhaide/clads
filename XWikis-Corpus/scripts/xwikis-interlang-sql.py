# ##
# Example of how to run the query , dewiki is a view that needs to be created in the DB
# cursor_ids.execute(get_ids_query, ("Arthur Wellesley, 1. Duke of Wellington", 'dewiki'))
# cursor_langs.execute(get_langs_query, (oo,))
#
# CREATE VIEW dewiki AS
#    select * from wb_items_per_site where ips_site_id='dewiki';
#
# ###


import sys, argparse
import mysql.connector

def run(args):
    cnx = mysql.connector.connect(user=args.user, database='wikidatawiki', password=args.password)
    cursor_ids = cnx.cursor()
    cursor_langs = cnx.cursor()

    get_ids_query = "SELECT ips_item_id FROM wb_items_per_site WHERE ips_site_page= %s AND ips_site_id= %s"
    get_langs_query = "SELECT ips_site_id, ips_site_page FROM wb_items_per_site WHERE ips_item_id = %s"

    pivotwiki = args.pivot
    pivotFile = open(args.pivot_urls, 'r')
    interlangFile  = open(args.pivot_urls+ '.lang', 'w')
    getLangs = args.target #should be something like this: ['frwiki', 'enwiki', 'cswiki', 'zhwiki', 'eswiki', 'ruwiki']

    missing = 0
    for line in pivotFile.readlines():
        page = line.split(" ||| ")[0]
        page = page.replace('&amp;', '&')

        cursor_ids.execute(get_ids_query, (page, pivotwiki))

        found = False
        for t in cursor_ids:
            cursor_langs.execute(get_langs_query, t)
            found = True
            break
        if not found:
            print(page)
            print("PAGE NOT FOUND")
            missing +=1

        langs = [line.strip('\n')]
        for tlang in cursor_langs:
            if tlang[0] in getLangs:
              langs.append(tlang[0][0:2] + "::" + tlang[1])
        interlangFile.write(" ||| ".join(langs) + "\n")
        interlangFile.flush()


    print(" * not found: {}".format(missing))

    interlangFile.close()
    cursor_langs.close()
    cursor_ids.close()
    cnx.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
       description=__doc__,
       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--pivot', help="pivot wikipedia (e.g., frwiki.)", required=True)
    parser.add_argument('--pivot-urls',
                        help="file of pivot wikipedia with titles-url pairs (e.g., xwikis/frwiki-20200620.urls.)",
                        required=True)
    parser.add_argument('--password', help="for connection to the mysql server.", required=True)
    parser.add_argument('--user', help="for connection to the mysql server.", required=True)
    parser.add_argument('--targets', nargs='+',
                        help='list of target wikis to align with (e.g., frwiki, enwiki, cswiki, zhwiki, eswiki, ruwiki)',
                        required=True)

    args = parser.parse_args(sys.argv[1:])

    run(args)