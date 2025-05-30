#!/usr/bin/env python3
"""
dir-report.py – Optimized directory inventory for AWS network storage

Usage:
    python dir-report.py [options]
    
Options:
    --target DIR        Root directory to analyze (default: .)
    --output DIR        Where to save .md and .csv (default: .)
    --depth N          Max depth to search (default: 2)
    --exclude LIST     Comma-separated dirs to skip (default: .git,node_modules)
    --sample N         Random sample size for large datasets
    --mode MODE        'tally' for quick counts, 'full' for detailed (default: full)
    --workers N        Number of parallel workers (default: 4)
    --batch-size N     Items to process per batch (default: 1000)
    --csv              Also save data as CSV
    --help, -h         Show this help
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime
import humanize
from collections import Counter, defaultdict
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from dataclasses import dataclass
from typing import List, Dict, Optional
import stat


@dataclass
class FileInfo:
    """Optimized file info structure"""
    path: str
    name: str
    parent: str
    suffix: str
    size_bytes: int
    modified: float
    is_file: bool
    is_dir: bool
    depth: int


class OptimizedDirectoryScanner:
    """
    Network-storage optimized directory scanner that minimizes system calls
    and uses parallel processing for better performance on AWS mounted volumes.
    """
    
    def __init__(self, max_workers=4, batch_size=1000, show_progress=True):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.stats = {
            'files': 0,
            'dirs': 0,
            'total': 0,
            'errors': 0
        }
        self.stats_lock = threading.Lock()
        
    def should_skip_dir(self, path_parts, exclude_dirs):
        """Check if directory should be skipped based on exclude list"""
        return any(part in exclude_dirs for part in path_parts)
    
    def batch_scan_directory(self, dir_path, max_depth, current_depth, exclude_dirs):
        """
        Scan a directory using os.scandir() for optimal network performance.
        This minimizes system calls compared to individual stat operations.
        """
        results = []
        subdirs_to_process = []
        
        try:
            # Use scandir for efficient directory listing with stat info
            with os.scandir(dir_path) as entries:
                for entry in entries:
                    try:
                        # Get relative path for exclusion checking
                        rel_path = Path(entry.path).relative_to(Path(dir_path).parent)
                        if self.should_skip_dir(rel_path.parts, exclude_dirs):
                            continue
                        
                        # Get stat info efficiently (scandir caches this)
                        stat_info = entry.stat(follow_symlinks=False)
                        
                        file_info = FileInfo(
                            path=entry.path,
                            name=entry.name,
                            parent=str(Path(entry.path).parent),
                            suffix=Path(entry.name).suffix.lower() if Path(entry.name).suffix else '<none>',
                            size_bytes=stat_info.st_size,
                            modified=stat_info.st_mtime,
                            is_file=stat.S_ISREG(stat_info.st_mode),
                            is_dir=stat.S_ISDIR(stat_info.st_mode),
                            depth=current_depth
                        )
                        
                        results.append(file_info)
                        
                        # Update stats
                        with self.stats_lock:
                            if file_info.is_file:
                                self.stats['files'] += 1
                            elif file_info.is_dir:
                                self.stats['dirs'] += 1
                            self.stats['total'] += 1
                        
                        # Collect subdirectories for further processing
                        if file_info.is_dir and current_depth < max_depth:
                            subdirs_to_process.append((entry.path, current_depth + 1))
                            
                    except (OSError, PermissionError):
                        with self.stats_lock:
                            self.stats['errors'] += 1
                        continue
                        
        except (OSError, PermissionError):
            with self.stats_lock:
                self.stats['errors'] += 1
            return results, subdirs_to_process
        
        return results, subdirs_to_process
    
    def scan_directory_parallel(self, target_path, max_depth=2, exclude_dirs=None, sample_size=None, mode='full'):
        """
        Parallel directory scanning optimized for network storage.
        """
        if exclude_dirs is None:
            exclude_dirs = {'.git', 'node_modules', '__pycache__', '.venv', '.DS_Store'}
        else:
            exclude_dirs = set(exclude_dirs)
        
        target = Path(target_path).resolve()
        all_results = []
        file_reservoir = []
        
        # Initialize progress tracking
        last_update = time.time()
        update_interval = 2.0  # Update every 2 seconds
        
        def update_progress():
            if self.show_progress:
                current_time = time.time()
                if current_time - last_update > update_interval:
                    if mode == 'tally':
                        print(f"\r  Files: {self.stats['files']:,} | Dirs: {self.stats['dirs']:,} | Total: {self.stats['total']:,}", 
                              end='', file=sys.stderr)
                    else:
                        print(f"\r  Processed: {self.stats['total']:,} (Files: {self.stats['files']:,}, Dirs: {self.stats['dirs']:,}, Errors: {self.stats['errors']:,})", 
                              end='', file=sys.stderr)
                    return current_time
            return last_update
        
        # Start with the target directory
        work_queue = [(str(target), 0)]
        
        # Use ThreadPoolExecutor for I/O bound operations
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while work_queue:
                # Submit batch of work
                current_batch = work_queue[:self.batch_size]
                work_queue = work_queue[self.batch_size:]
                
                # Submit scanning tasks
                future_to_dir = {
                    executor.submit(self.batch_scan_directory, dir_path, max_depth, depth, exclude_dirs): (dir_path, depth)
                    for dir_path, depth in current_batch
                }
                
                # Collect results
                for future in as_completed(future_to_dir):
                    try:
                        results, subdirs = future.result()
                        all_results.extend(results)
                        work_queue.extend(subdirs)
                        
                        # Handle sampling for files
                        if sample_size:
                            for item in results:
                                if item.is_file:
                                    if len(file_reservoir) < sample_size:
                                        file_reservoir.append(item)
                                    else:
                                        j = random.randint(0, self.stats['files'] - 1)
                                        if j < sample_size:
                                            file_reservoir[j] = item
                        
                        # Update progress
                        update_progress()
                        
                    except Exception as e:
                        print(f"\nError processing directory: {e}", file=sys.stderr)
                        with self.stats_lock:
                            self.stats['errors'] += 1
        
        # Final progress update
        if self.show_progress:
            if mode == 'tally':
                print(f"\r  Files: {self.stats['files']:,} | Dirs: {self.stats['dirs']:,} | Total: {self.stats['total']:,}", flush=True)
            else:
                print(f"\r  Processed: {self.stats['total']:,} (Files: {self.stats['files']:,}, Dirs: {self.stats['dirs']:,}, Errors: {self.stats['errors']:,})", flush=True)
            print()  # New line
        
        # Convert to DataFrame
        if all_results:
            data = []
            for item in all_results:
                data.append({
                    'path': item.path,
                    'name': item.name,
                    'parent': item.parent,
                    'suffix': item.suffix,
                    'size_bytes': item.size_bytes,
                    'size_human': humanize.naturalsize(item.size_bytes),
                    'modified': datetime.fromtimestamp(item.modified),
                    'is_file': item.is_file,
                    'is_dir': item.is_dir,
                    'depth_from_target': item.depth,
                    'relative_path': str(Path(item.path).relative_to(target)) if item.path != str(target) else '.'
                })
            
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame()
        
        # Store sampling info
        if sample_size and file_reservoir:
            sampled_data = []
            for item in file_reservoir:
                sampled_data.append({
                    'path': item.path,
                    'suffix': item.suffix,
                    'size_bytes': item.size_bytes
                })
            df.attrs['sampled_files'] = pd.DataFrame(sampled_data)
            df.attrs['is_sampled'] = True
            df.attrs['sample_size'] = len(file_reservoir)
        else:
            df.attrs['is_sampled'] = False
        
        return df


def generate_tally_report(df, target_path):
    """Generate quick tally report."""
    target = Path(target_path)
    
    # Basic counts
    total_size = sum(df[df['is_file']]['size_bytes']) if not df.empty else 0
    files = df[df['is_file']]
    dirs = df[df['is_dir']]
    
    # Top-level items (depth 1 from target)
    top_level = df[df['depth_from_target'] == 1] if not df.empty else pd.DataFrame()
    
    report = f"""## Quick tally report for `{target}` ({datetime.now().strftime('%Y-%m-%d %H:%M')})

- Total size on disk: **{humanize.naturalsize(total_size)}**
- Top-level items: **{len(top_level)}**
- Files (≤depth): **{len(files)}**
- Directories (≤depth): **{len(dirs)}**
"""
    return report


def generate_full_report(df, target_path):
    """Generate detailed report with breakdowns."""
    target = Path(target_path)
    
    # Basic stats
    files = df[df['is_file']]
    dirs = df[df['is_dir']]
    total_size = files['size_bytes'].sum() if not files.empty else 0
    top_level = df[df['depth_from_target'] == 1] if not df.empty else pd.DataFrame()
    
    report = f"""## Directory report for `{target}` ({datetime.now().strftime('%Y-%m-%d %H:%M')})

### Overview
- Total size on disk: **{humanize.naturalsize(total_size)}**
- Top-level items: **{len(top_level)}**
- Regular files (≤depth): **{len(files)}**
- Directories (≤depth): **{len(dirs)}**
"""
    
    # Size breakdown of top-level items
    if not top_level.empty:
        report += "\n### Size of each first-level entry\n\n"
        
        # Calculate sizes for top-level items efficiently
        top_level_sizes = []
        for _, item in top_level.iterrows():
            if item['is_dir']:
                # Sum sizes of all files in this directory
                dir_path = item['relative_path']
                if dir_path != '.':
                    dir_files = files[files['relative_path'].str.startswith(dir_path + '/')]
                    size = dir_files['size_bytes'].sum()
                else:
                    size = item['size_bytes']
            else:
                size = item['size_bytes']
            
            top_level_sizes.append({
                'name': item['name'],
                'size_bytes': size,
                'size_human': humanize.naturalsize(size)
            })
        
        # Sort by size and display
        top_level_sizes.sort(key=lambda x: x['size_bytes'], reverse=True)
        for item in top_level_sizes:
            report += f"- {item['size_human']}  {item['name']}\n"
        report += "\n"
    
    # File type breakdown
    if not files.empty:
        # Use sampled data if available
        analysis_data = df.attrs.get('sampled_files', files)
        if df.attrs.get('is_sampled', False):
            report += f"### File-type breakdown (sampled from {df.attrs['sample_size']} files)\n\n"
        else:
            report += "### File-type breakdown\n\n"
        
        ext_counts = Counter(analysis_data['suffix'])
        for ext, count in ext_counts.most_common():
            report += f"- **{ext}**: {count} files\n"
    
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--target', default='.', help='Root directory to analyze')
    parser.add_argument('--output', default='.', help='Where to save output files')
    parser.add_argument('--depth', type=int, default=2, help='Max depth to search')
    parser.add_argument('--exclude', help='Comma-separated dirs to skip')
    parser.add_argument('--sample', type=int, help='Random sample size for large datasets')
    parser.add_argument('--mode', choices=['tally', 'full'], default='full', help='Report mode')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--batch-size', type=int, default=1000, help='Items to process per batch')
    parser.add_argument('--csv', action='store_true', help='Also save data as CSV')
    
    args = parser.parse_args()
    
    # Validate paths
    target_path = Path(args.target)
    if not target_path.exists():
        print(f"Error: Target '{target_path}' not found", file=sys.stderr)
        sys.exit(1)
    
    output_path = Path(args.output)
    if not output_path.exists():
        print(f"Error: Output directory '{output_path}' not found", file=sys.stderr)
        sys.exit(1)
    
    # Parse exclusions
    exclude_dirs = None
    if args.exclude:
        exclude_dirs = [d.strip() for d in args.exclude.split(',')]
    
    # Initialize scanner with optimization for network storage
    scanner = OptimizedDirectoryScanner(
        max_workers=args.workers,
        batch_size=args.batch_size,
        show_progress=True
    )
    
    # Scan directory with optimizations
    print(f"Scanning directory tree (mode: {args.mode}, workers: {args.workers})...")
    start_time = time.time()
    
    df = scanner.scan_directory_parallel(
        target_path, 
        max_depth=args.depth,
        exclude_dirs=exclude_dirs,
        sample_size=args.sample,
        mode=args.mode
    )
    
    scan_time = time.time() - start_time
    
    # Final summary
    files_count = len(df[df['is_file']]) if not df.empty else 0
    dirs_count = len(df[df['is_dir']]) if not df.empty else 0
    print(f"Scan complete! Files: {files_count:,} | Directories: {dirs_count:,} | Total: {len(df):,} | Time: {scan_time:.1f}s")
    
    # Generate report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    target_name = target_path.name if target_path.name else 'root'
    
    if args.mode == 'tally':
        report_content = generate_tally_report(df, target_path)
    else:
        report_content = generate_full_report(df, target_path)
    
    # Save markdown report
    report_file = output_path / f"{target_name}-report-{timestamp}.md"
    with open(report_file, 'w') as f:
        f.write(report_content)
    print(f"Markdown report written to: {report_file}")
    
    # Save CSV if requested
    if args.csv and not df.empty:
        csv_file = output_path / f"{target_name}-data-{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"CSV data written to: {csv_file}")
    
    # Return the DataFrame for interactive use
    return df


if __name__ == '__main__':
    df = main()
