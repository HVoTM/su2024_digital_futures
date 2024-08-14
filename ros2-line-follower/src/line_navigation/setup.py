import os
from glob import glob
from setuptools import setup

package_name = 'line_navigation'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
        (os.path.join('share', package_name), glob('worlds/*.world')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hungvo',
    maintainer_email='voht@mail.uc.edu',
    description='Operating Line detection and following features for TurtleBot3',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'follower_node = line_navigation.line_follower:main',
        	'follower_w_visual = line_navigation.line_follower_display:main',
            'obstacle_avoidance = line_navigation.obstacle_avoidance:main',
            'crazy_wanderer = line_navigation.wandering_random_color:main',
        ],
    },
)
