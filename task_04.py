import os
import shutil
from datetime import datetime
import logging
from pathlib import Path

class FileOrganizer:
    """A class to organize files in a directory based on their extensions."""
    
    def __init__(self, source_dir):
        """Initialize with source directory and setup logging."""
        self.source_dir = Path(source_dir)
        self.organized_dir = self.source_dir / "organized_files"
        
        # Logging
        logging.basicConfig(
            filename='file_organizer.log',
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Extension categories
        self.categories = {
            'documents': ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt'],
            'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
            'audio': ['.mp3', '.wav', '.flac', '.m4a'],
            'videos': ['.mp4', '.avi', '.mkv', '.mov'],
            'archives': ['.zip', '.rar', '.7z', '.tar', '.gz'],
            'code': ['.py', '.js', '.html', '.css', '.java', '.cpp']
        }

    def create_directories(self):
        """Create organized directories if they don't exist."""
        try:
            if not self.organized_dir.exists():
                self.organized_dir.mkdir()
            
            for category in self.categories:
                category_dir = self.organized_dir / category
                if not category_dir.exists():
                    category_dir.mkdir()
            
            # Create misc directory for uncategorized files
            misc_dir = self.organized_dir / 'misc'
            if not misc_dir.exists():
                misc_dir.mkdir()
                
            logging.info("Directory structure created successfully")
        except Exception as e:
            logging.error(f"Error creating directories: {str(e)}")
            raise

    def get_category(self, file_extension):
        """Determine the category of a file based on its extension."""
        for category, extensions in self.categories.items():
            if file_extension.lower() in extensions:
                return category
        return 'misc'

    def organize_files(self):
        """Organize files from source directory into appropriate categories."""
        try:
            total_files = 0
            organized_files = 0
            
            # Required directories
            self.create_directories()
            
            # Iterate through all files source directory
            for file_path in self.source_dir.iterdir():
                if file_path.is_file() and file_path != Path('file_organizer.log'):
                    total_files += 1
                    
                    # Get file extension 
                    file_extension = file_path.suffix
                    category = self.get_category(file_extension)
                    
                    # Create destination path
                    dest_dir = self.organized_dir / category
                    dest_path = dest_dir / file_path.name
                    
                    # Handle duplicate files
                    if dest_path.exists():
                        base_name = dest_path.stem
                        extension = dest_path.suffix
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        dest_path = dest_dir / f"{base_name}_{timestamp}{extension}"
                    
                    # Move the file
                    shutil.move(str(file_path), str(dest_path))
                    organized_files += 1
                    logging.info(f"Moved '{file_path.name}' to {category} folder")
            
            # Log summary
            logging.info(f"Organization complete. Processed {organized_files} of {total_files} files")
            return organized_files, total_files
            
        except Exception as e:
            logging.error(f"Error organizing files: {str(e)}")
            raise

    def generate_report(self):
        """Generate a report of the organization results."""
        report = []
        try:
            report.append("File Organization Report")
            report.append("=" * 25)
            
            for category in self.categories.keys():
                category_dir = self.organized_dir / category
                files = list(category_dir.glob('*'))
                report.append(f"{category.capitalize()}: {len(files)} files")
            
            # Count misc files
            misc_dir = self.organized_dir / 'misc'
            misc_files = list(misc_dir.glob('*'))
            report.append(f"Miscellaneous: {len(misc_files)} files")
            
            # Write report to file
            report_path = self.organized_dir / 'organization_report.txt'
            report_path.write_text('\n'.join(report))
            logging.info("Report generated successfully")
            
        except Exception as e:
            logging.error(f"Error generating report: {str(e)}")
            raise

def main():
    """Main function to run the file organizer."""
    try:
        # Get source directory from user
        source_dir = input("Enter the directory path to organize: ")
        
        # Create and run 
        organizer = FileOrganizer(source_dir)
        organized, total = organizer.organize_files()
        organizer.generate_report()
        
        print(f"\nOrganization complete!")
        print(f"Processed {organized} of {total} files")
        print(f"Check 'organization_report.txt' in the organized_files directory for details")
        print(f"Check 'file_organizer.log' for detailed operation logs")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logging.error(f"Program terminated with error: {str(e)}")

if __name__ == "__main__":
    main()