#!/usr/bin/perl

use strict;
use warnings;
use LWP::Simple;

#download bam files
#convert bam files to bed (keep both)
#run MACS2 and JAMM
#export
#delete both

my $baseURL = 'http://hgdownload.soe.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/';
my $page=get($baseURL);

my @fileNames = split(/\">/, $page);

my $maxFiles = 10000;
my $countFiles = 1;

foreach my $file (@fileNames) {

	if($countFiles <= $maxFiles) {
		my @split = split(/<\/a>/, $file);
		my $file = $split[0];

		if($file =~ /\.bam/ && !($file =~ /\.bam\.ba/)) {
			$countFiles++;
			my $url = $baseURL . $file;

			print "downloading ".$file."...\n";
			getstore($url, $file);

			my @nameSplit = split(/\./, $file);
			my $outputName = $nameSplit[0];

			print "MACS2 peak calling for ".$file."...\n";
			my $callPeakOutput = `macs2 callpeak -t $file -f BAM -g hs -n macs2Output/$outputName`;

			print "deleting ".$file."...\n";
			my $removeFileOutput = `rm $file`;
		}
	}
}
