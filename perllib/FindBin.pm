package FindBin;
use strict; use Cwd qw(abs_path); use File::Basename;
our ($Bin, $Script, $RealBin, $RealScript, $Dir, $RealDir);
sub again { init(); }
sub init {
    my $script = $0;
    my $abs = abs_path($script);
    $abs = $script unless defined $abs;
    $RealScript = $Script = basename($abs);
    $RealBin = $Bin = dirname($abs);
    $RealDir = $Dir = $Bin;
}
init();
our @EXPORT_OK = qw($Bin $Script $RealBin $RealScript $Dir $RealDir);
require Exporter; our @ISA = qw(Exporter);
1;
