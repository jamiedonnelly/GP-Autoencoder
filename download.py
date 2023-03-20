import cdsapi
import sys

def download(month):
    c.retrieve(
    'efas-historical',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            '2m_temperature', 'surface_pressure',
        ],
        'year': '2022',
        'time': [
            '00:00', '03:00', '06:00',
            '09:00', '12:00', '15:00',
            '18:00', '21:00',
        ],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'month': month,
    },
    f'month{month}.nc')

def main(month):
    global c
    c = cdsapi.Client()
    download(month)


if __name__=="__main__":
    month = sys.argv[1]
    main(month)
