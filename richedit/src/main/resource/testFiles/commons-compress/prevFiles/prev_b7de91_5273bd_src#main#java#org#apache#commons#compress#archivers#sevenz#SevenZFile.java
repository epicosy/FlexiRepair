/*
 *  Licensed to the Apache Software Foundation (ASF) under one or more
 *  contributor license agreements.  See the NOTICE file distributed with
 *  this work for additional information regarding copyright ownership.
 *  The ASF licenses this file to You under the Apache License, Version 2.0
 *  (the "License"); you may not use this file except in compliance with
 *  the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */
package org.apache.commons.compress.archivers.sevenz;

import java.io.ByteArrayInputStream;
import java.io.DataInput;
import java.io.DataInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.RandomAccessFile;
import java.util.Arrays;
import java.util.BitSet;
import java.util.zip.CRC32;

import org.apache.commons.compress.utils.CRC32VerifyingInputStream;

/**
 * Reads a 7z file, using RandomAccessFile under
 * the covers.
 * <p>
 * The 7z file format is a flexible container
 * that can contain many compression and
 * encryption types, but at the moment only
 * only Copy, LZMA2, BZIP2, and AES-256 + SHA-256
 * are supported, and archive header compression
 * (when it uses the unsupported LZMA
 * compression) isn't. So the only archives
 * that can be read are the following:
 * <pre>
 * 7z a -mhc=off [-mhe=on] -mx=0 [-ppassword] archive.7z files
 * 7z a -mhc=off [-mhe=on] -m0=LZMA2 [-ppassword] archive.7z files
 * 7z a -mhc=off [-mhe=on] -m0=BZIP2 [-ppassword] archive.7z files
 * </pre>
 * <p>
 * The format is very Windows/Intel specific,
 * so it uses little-endian byte order,
 * doesn't store user/group or permission bits,
 * and represents times using NTFS timestamps
 * (100 nanosecond units since 1 January 1601).
 * Hence the official tools recommend against
 * using it for backup purposes on *nix, and
 * recommend .tar.7z or .tar.lzma or .tar.xz
 * instead.  
 * <p>
 * Both the header and file contents may be
 * compressed and/or encrypted. With both
 * encrypted, neither file names nor file
 * contents can be read, but the use of
 * encryption isn't plausibly deniable.
 * 
 * @NotThreadSafe
 */
public class SevenZFile {
    private static final boolean DEBUG = false;
    private static final int SIGNATURE_HEADER_SIZE = 32;
    private RandomAccessFile file;
    private final Archive archive;
    private int currentEntryIndex = -1;
    private int currentFolderIndex = -1;
    private InputStream currentFolderInputStream = null;
    private InputStream currentEntryInputStream = null;
    private String password;
        
    private static final byte[] sevenZSignature = {
        (byte)'7', (byte)'z', (byte)0xBC, (byte)0xAF, (byte)0x27, (byte)0x1C
    };
    
    public SevenZFile(final File filename, final String password) throws IOException {
        boolean succeeded = false;
        this.password = password;
        this.file = new RandomAccessFile(filename, "r");
        try {
            archive = readHeaders();
            succeeded = true;
        } finally {
            if (!succeeded) {
                this.file.close();
            }
        }
    }
    
    public SevenZFile(final File filename) throws IOException {
        this(filename, null);
    }

    public void close() {
        if (file != null) {
            try {
                file.close();
            } catch (IOException ignored) { // NOPMD
            }
            file = null;
        }
    }
    
    private static void debug(String str) {
        if (DEBUG) {
            System.out.println(str);
        }
    }
    
    private static void debug(String fmt, Object... args) {
        if (DEBUG) {
            System.out.format(fmt, args);
        }
    }
    
    public SevenZArchiveEntry getNextEntry() throws IOException {
        if (currentEntryIndex >= (archive.files.length - 1)) {
            return null;
        }
        ++currentEntryIndex;
        final SevenZArchiveEntry entry = archive.files[currentEntryIndex];
        buildDecodingStream();
        return entry;
    }
    
    private Archive readHeaders() throws IOException {
        debug("SignatureHeader");
        
        final byte[] signature = new byte[6];
        file.readFully(signature);
        if (!Arrays.equals(signature, sevenZSignature)) {
            throw new IOException("Bad 7z signature");
        }
        // 7zFormat.txt has it wrong - it's first major then minor
        final byte archiveVersionMajor = file.readByte();
        final byte archiveVersionMinor = file.readByte();
        debug("  archiveVersion major=%d, minor=%d\n",
                archiveVersionMajor, archiveVersionMinor);
        if (archiveVersionMajor != 0) {
            throw new IOException(String.format("Unsupported 7z version (%d,%d)",
                    archiveVersionMajor, archiveVersionMinor));
        }

        final int startHeaderCrc = Integer.reverseBytes(file.readInt());
        final StartHeader startHeader = readStartHeader(startHeaderCrc);
        
        final int nextHeaderSizeInt = (int) startHeader.nextHeaderSize;
        if (nextHeaderSizeInt != startHeader.nextHeaderSize) {
            throw new IOException("cannot handle nextHeaderSize " + startHeader.nextHeaderSize);
        }
        file.seek(SIGNATURE_HEADER_SIZE + startHeader.nextHeaderOffset);
        final byte[] nextHeader = new byte[nextHeaderSizeInt];
        file.readFully(nextHeader);
        final CRC32 crc = new CRC32();
        crc.update(nextHeader);
        if (startHeader.nextHeaderCrc != (int) crc.getValue()) {
            throw new IOException("NextHeader CRC mismatch");
        }
        
        final ByteArrayInputStream byteStream = new ByteArrayInputStream(nextHeader);
        DataInputStream nextHeaderInputStream = new DataInputStream(
                byteStream);
        Archive archive = new Archive();
        int nid = nextHeaderInputStream.readUnsignedByte();
        if (nid == NID.kEncodedHeader) {
            nextHeaderInputStream = readEncodedHeader(nextHeaderInputStream, archive);
            // Archive gets rebuilt with the new header
            archive = new Archive();
            nid = nextHeaderInputStream.readUnsignedByte();
        }
        if (nid == NID.kHeader) {
            readHeader(nextHeaderInputStream, archive);
        } else {
            throw new IOException("Broken or unsupported archive: no Header");
        }
        return archive;
    }
    
    private StartHeader readStartHeader(final int startHeaderCrc) throws IOException {
        final StartHeader startHeader = new StartHeader();
        DataInputStream dataInputStream = null;
        try {
             dataInputStream = new DataInputStream(new CRC32VerifyingInputStream(
                    new BoundedRandomAccessFileInputStream(file, 20), 20, startHeaderCrc));
             startHeader.nextHeaderOffset = Long.reverseBytes(dataInputStream.readLong());
             startHeader.nextHeaderSize = Long.reverseBytes(dataInputStream.readLong());
             startHeader.nextHeaderCrc = Integer.reverseBytes(dataInputStream.readInt());
             return startHeader;
        } finally {
            if (dataInputStream != null) {
                dataInputStream.close();
            }
        }
    }
    
    private void readHeader(final DataInput header, final Archive archive) throws IOException {
        debug("Header");

        int nid = header.readUnsignedByte();
        
        if (nid == NID.kArchiveProperties) {
            readArchiveProperties(header);
            nid = header.readUnsignedByte();
        }
        
        if (nid == NID.kAdditionalStreamsInfo) {
            throw new IOException("Additional streams unsupported");
            //nid = header.readUnsignedByte();
        }
        
        if (nid == NID.kMainStreamsInfo) {
            readStreamsInfo(header, archive);
            nid = header.readUnsignedByte();
        }
        
        if (nid == NID.kFilesInfo) {
            readFilesInfo(header, archive);
            nid = header.readUnsignedByte();
        }
        
        if (nid != NID.kEnd) {
            throw new IOException("Badly terminated header");
        }
    }
    
    private void readArchiveProperties(final DataInput input) throws IOException {
        // FIXME: the reference implementation just throws them away?
        debug("ArchiveProperties");

        int nid =  input.readUnsignedByte();
        while (nid != NID.kEnd) {
            final long propertySize = readUint64(input);
            final byte[] property = new byte[(int)propertySize];
            input.readFully(property);
            nid = input.readUnsignedByte();
        }
    }
    
    private DataInputStream readEncodedHeader(final DataInputStream header, final Archive archive) throws IOException {
        debug("EncodedHeader");

        readStreamsInfo(header, archive);
        
        // FIXME: merge with buildDecodingStream()/buildDecoderStack() at some stage?
        final Folder folder = archive.folders[0];
        final int firstPackStreamIndex = 0;
        final long folderOffset = SIGNATURE_HEADER_SIZE + archive.packPos +
                0;
        
        file.seek(folderOffset);
        InputStream inputStreamStack = new BoundedRandomAccessFileInputStream(file,
                archive.packSizes[firstPackStreamIndex]);
        for (final Coder coder : folder.coders) {
            if (coder.numInStreams != 1 || coder.numOutStreams != 1) {
                throw new IOException("Multi input/output stream coders are not yet supported");
            }
            inputStreamStack = Coders.addDecoder(inputStreamStack, coder, password);
        }
        if (folder.hasCrc) {
            inputStreamStack = new CRC32VerifyingInputStream(inputStreamStack,
                    folder.getUnpackSize(), folder.crc);
        }
        final byte[] nextHeader = new byte[(int)folder.getUnpackSize()];
        final DataInputStream nextHeaderInputStream = new DataInputStream(inputStreamStack);
        try {
            nextHeaderInputStream.readFully(nextHeader);
        } finally {
            nextHeaderInputStream.close();
        }
        return new DataInputStream(new ByteArrayInputStream(nextHeader));

        
        //throw new IOException("LZMA compression unsupported, so files with compressed header cannot be read");
        // FIXME: this extracts the header to an LZMA file which can then be
        // manually decompressed.
//        long offset = SIGNATURE_HEADER_SIZE + archive.packPos;
//        file.seek(offset);
//        long unpackSize = archive.folders[0].getUnpackSize();
//        byte[] packed = new byte[(int)archive.packSizes[0]];
//        file.readFully(packed);
//        
//        FileOutputStream fos = new FileOutputStream(new File("/tmp/encodedHeader.7z"));
//        fos.write(archive.folders[0].coders[0].properties);
//        // size - assuming < 256
//        fos.write((int)(unpackSize & 0xff));
//        fos.write(0);
//        fos.write(0);
//        fos.write(0);
//        fos.write(0);
//        fos.write(0);
//        fos.write(0);
//        fos.write(0);
//        fos.write(packed);
//        fos.close();
    }
    
    private void readStreamsInfo(final DataInput header, final Archive archive) throws IOException {
        debug("StreamsInfo");
        
        int nid = header.readUnsignedByte();
        
        if (nid == NID.kPackInfo) {
            readPackInfo(header, archive);
            nid = header.readUnsignedByte();
        }
        
        if (nid == NID.kUnpackInfo) {
            readUnpackInfo(header, archive);
            nid = header.readUnsignedByte();
        }
        
        if (nid == NID.kSubStreamsInfo) {
            readSubStreamsInfo(header, archive);
            nid = header.readUnsignedByte();
        }
        
        if (nid != NID.kEnd) {
            throw new IOException("Badly terminated StreamsInfo");
        }
    }
    
    private void readPackInfo(final DataInput header, final Archive archive) throws IOException {
        debug("PackInfo");
        
        archive.packPos = readUint64(header);
        final long numPackStreams = readUint64(header);
        debug("  " + numPackStreams + " pack streams");
        
        int nid = header.readUnsignedByte();
        if (nid == NID.kSize) {
            archive.packSizes = new long[(int)numPackStreams];
            for (int i = 0; i < archive.packSizes.length; i++) {
                archive.packSizes[i] = readUint64(header);
                debug("  pack size %d is %d\n", i, archive.packSizes[i]);
            }
            nid = header.readUnsignedByte();
        }
        
        if (nid == NID.kCRC) {
            archive.packCrcsDefined = readAllOrBits(header, (int)numPackStreams);
            archive.packCrcs = new int[(int)numPackStreams];
            for (int i = 0; i < (int)numPackStreams; i++) {
                if (archive.packCrcsDefined.get(i)) {
                    archive.packCrcs[i] = Integer.reverseBytes(header.readInt());
                }
            }
            
            nid = header.readUnsignedByte();
        }
        
        if (nid != NID.kEnd) {
            throw new IOException("Badly terminated PackInfo (" + nid + ")");
        }
    }
    
    private void readUnpackInfo(final DataInput header, final Archive archive) throws IOException {
        debug("UnpackInfo");

        int nid = header.readUnsignedByte();
        if (nid != NID.kFolder) {
            throw new IOException("Expected kFolder, got " + nid);
        }
        final long numFolders = readUint64(header);
        debug("  " + numFolders + " folders");
        final Folder[] folders = new Folder[(int)numFolders];
        archive.folders = folders;
        final int external = header.readUnsignedByte();
        if (external != 0) {
            throw new IOException("External unsupported");
        } else {
            for (int i = 0; i < (int)numFolders; i++) {
                folders[i] = readFolder(header);
            }
        }
        
        nid = header.readUnsignedByte();
        if (nid != NID.kCodersUnpackSize) {
            throw new IOException("Expected kCodersUnpackSize, got " + nid);
        }
        for (final Folder folder : folders) {
            folder.unpackSizes = new long[(int)folder.totalOutputStreams];
            for (int i = 0; i < folder.totalOutputStreams; i++) {
                folder.unpackSizes[i] = readUint64(header);
            }
        }
        
        nid = header.readUnsignedByte();
        if (nid == NID.kCRC) {
            final BitSet crcsDefined = readAllOrBits(header, (int)numFolders);
            for (int i = 0; i < (int)numFolders; i++) {
                if (crcsDefined.get(i)) {
                    folders[i].hasCrc = true;
                    folders[i].crc = Integer.reverseBytes(header.readInt());
                } else {
                    folders[i].hasCrc = false;
                }
            }
            
            nid = header.readUnsignedByte();
        }
        
        if (nid != NID.kEnd) {
            throw new IOException("Badly terminated UnpackInfo");
        }
    }
    
    private void readSubStreamsInfo(final DataInput header, final Archive archive) throws IOException {
        debug("SubStreamsInfo");
        
        for (final Folder folder : archive.folders) {
            folder.numUnpackSubStreams = 1;
        }
        int totalUnpackStreams = archive.folders.length;
        
        int nid = header.readUnsignedByte();
        if (nid == NID.kNumUnpackStream) {
            totalUnpackStreams = 0;
            for (final Folder folder : archive.folders) {
                final long numStreams = readUint64(header);
                folder.numUnpackSubStreams = (int)numStreams;
                totalUnpackStreams += numStreams;
            }
            nid = header.readUnsignedByte();
        }
        
        final SubStreamsInfo subStreamsInfo = new SubStreamsInfo();
        subStreamsInfo.unpackSizes = new long[totalUnpackStreams];
        subStreamsInfo.hasCrc = new BitSet(totalUnpackStreams);
        subStreamsInfo.crcs = new int[totalUnpackStreams];
        
        int nextUnpackStream = 0;
        for (final Folder folder : archive.folders) {
            if (folder.numUnpackSubStreams == 0) {
                continue;
            }
            long sum = 0;
            if (nid == NID.kSize) {
                for (int i = 0; i < (folder.numUnpackSubStreams - 1); i++) {
                    final long size = readUint64(header);
                    subStreamsInfo.unpackSizes[nextUnpackStream++] = size;
                    sum += size;
                }
            }
            subStreamsInfo.unpackSizes[nextUnpackStream++] = folder.getUnpackSize() - sum;
        }
        if (nid == NID.kSize) {
            nid = header.readUnsignedByte();
        }
        
        int numDigests = 0;
        for (final Folder folder : archive.folders) {
            if (folder.numUnpackSubStreams != 1 || !folder.hasCrc) {
                numDigests += folder.numUnpackSubStreams;
            }
        }
        
        if (nid == NID.kCRC) {
            final BitSet hasMissingCrc = readAllOrBits(header, numDigests);
            final int[] missingCrcs = new int[numDigests];
            for (int i = 0; i < numDigests; i++) {
                if (hasMissingCrc.get(i)) {
                    missingCrcs[i] = Integer.reverseBytes(header.readInt());
                }
            }
            int nextCrc = 0;
            int nextMissingCrc = 0;
            for (final Folder folder: archive.folders) {
                if (folder.numUnpackSubStreams == 1 && folder.hasCrc) {
                    subStreamsInfo.hasCrc.set(nextCrc, true);
                    subStreamsInfo.crcs[nextCrc] = folder.crc;
                    ++nextCrc;
                } else {
                    for (int i = 0; i < folder.numUnpackSubStreams; i++) {
                        subStreamsInfo.hasCrc.set(nextCrc, hasMissingCrc.get(nextMissingCrc));
                        subStreamsInfo.crcs[nextCrc] = missingCrcs[nextMissingCrc];
                        ++nextCrc;
                        ++nextMissingCrc;
                    }
                }
            }
            
            nid = header.readUnsignedByte();
        }
        
        if (nid != NID.kEnd) {
            throw new IOException("Badly terminated SubStreamsInfo");
        }
        
        archive.subStreamsInfo = subStreamsInfo;
    }
    
    private Folder readFolder(final DataInput header) throws IOException {
        final Folder folder = new Folder();
        
        final long numCoders = readUint64(header);
        final Coder[] coders = new Coder[(int)numCoders];
        long totalInStreams = 0;
        long totalOutStreams = 0;
        for (int i = 0; i < coders.length; i++) {
            coders[i] = new Coder();
            int bits = header.readUnsignedByte();
            final int idSize = bits & 0xf;
            final boolean isSimple = ((bits & 0x10) == 0);
            final boolean hasAttributes = ((bits & 0x20) != 0);
            final boolean moreAlternativeMethods = ((bits & 0x80) != 0);
            
            coders[i].decompressionMethodId = new byte[idSize];
            header.readFully(coders[i].decompressionMethodId);
            if (isSimple) {
                coders[i].numInStreams = 1;
                coders[i].numOutStreams = 1;
            } else {
                coders[i].numInStreams = readUint64(header);
                coders[i].numOutStreams = readUint64(header);
            }
            totalInStreams += coders[i].numInStreams;
            totalOutStreams += coders[i].numOutStreams;
            if (hasAttributes) {
                final long propertiesSize = readUint64(header);
                coders[i].properties = new byte[(int)propertiesSize];
                header.readFully(coders[i].properties);
            }
            if (DEBUG) {
                final StringBuilder methodStr = new StringBuilder();
                for (final byte b : coders[i].decompressionMethodId) {
                    methodStr.append(String.format("%02X", 0xff & b));
                }
                debug("  coder entry %d numInStreams=%d, numOutStreams=%d, method=%s, properties=%s\n", i,
                        coders[i].numInStreams, coders[i].numOutStreams, methodStr.toString(),
                        Arrays.toString(coders[i].properties));
            }
            // would need to keep looping as above:
            while (moreAlternativeMethods) {
                throw new IOException("Alternative methods are unsupported, please report. " +
                        "The reference implementation doesn't support them either.");
            }
        }
        folder.coders = coders;
        folder.totalInputStreams = totalInStreams;
        folder.totalOutputStreams = totalOutStreams;
        
        if (totalOutStreams == 0) {
            throw new IOException("Total output streams can't be 0");
        }
        final long numBindPairs = totalOutStreams - 1;
        final BindPair[] bindPairs = new BindPair[(int)numBindPairs];
        for (int i = 0; i < bindPairs.length; i++) {
            bindPairs[i] = new BindPair();
            bindPairs[i].inIndex = readUint64(header);
            bindPairs[i].outIndex = readUint64(header);
            debug("  bind pair in=%d out=%d\n", bindPairs[i].inIndex, bindPairs[i].outIndex);
        }
        folder.bindPairs = bindPairs;
        
        if (totalInStreams < numBindPairs) {
            throw new IOException("Total input streams can't be less than the number of bind pairs");
        }
        final long numPackedStreams = totalInStreams - numBindPairs;
        final long packedStreams[] = new long[(int)numPackedStreams];
        if (numPackedStreams == 1) {
            int i;
            for (i = 0; i < (int)totalInStreams; i++) {
                if (folder.findBindPairForInStream(i) < 0) {
                    break;
                }
            }
            if (i == (int)totalInStreams) {
                throw new IOException("Couldn't find stream's bind pair index");
            }
            packedStreams[0] = i;
        } else {
            for (int i = 0; i < (int)numPackedStreams; i++) {
                packedStreams[i] = readUint64(header);
            }
        }
        folder.packedStreams = packedStreams;
        
        return folder;
    }
    
    private BitSet readAllOrBits(final DataInput header, final int size) throws IOException {
        final int areAllDefined = header.readUnsignedByte();
        final BitSet bits;
        if (areAllDefined != 0) {
            bits = new BitSet(size);
            for (int i = 0; i < size; i++) {
                bits.set(i, true);
            }
        } else {
            bits = readBits(header, size);
        }
        return bits;
    }
    
    private BitSet readBits(final DataInput header, final int size) throws IOException {
        final BitSet bits = new BitSet(size);
        int mask = 0;
        int cache = 0;
        for (int i = 0; i < size; i++) {
            if (mask == 0) {
                mask = 0x80;
                cache = header.readUnsignedByte();
            }
            bits.set(i, (cache & mask) != 0);
            mask >>>= 1;
        }
        return bits;
    }
    
    private void readFilesInfo(final DataInput header, final Archive archive) throws IOException {
        debug("FilesInfo");

        final long numFiles = readUint64(header);
        final SevenZArchiveEntry[] files = new SevenZArchiveEntry[(int)numFiles];
        for (int i = 0; i < files.length; i++) {
            files[i] = new SevenZArchiveEntry();
        }
        BitSet isEmptyStream = null;
        BitSet isEmptyFile = null; 
        BitSet isAnti = null;
        while (true) {
            final int propertyType = header.readUnsignedByte();
            if (propertyType == 0) {
                break;
            }
            long size = readUint64(header);
            switch (propertyType) {
                case NID.kEmptyStream: {
                    debug("  kEmptyStream");
                    isEmptyStream = readBits(header, files.length);
                    break;
                }
                case NID.kEmptyFile: {
                    debug("  kEmptyFile");
                    if (isEmptyStream == null) { // protect against NPE
                        throw new IOException("Header format error: kEmptyStream must appear before kEmptyFile");
                    }
                    isEmptyFile = readBits(header, isEmptyStream.cardinality());
                    break;
                }
                case NID.kAnti: {
                    debug("  kAnti");
                    if (isEmptyStream == null) { // protect against NPE
                        throw new IOException("Header format error: kEmptyStream must appear before kAnti");
                    }
                    isAnti = readBits(header, isEmptyStream.cardinality());
                    break;
                }
                case NID.kName: {
                    debug("  kNames");
                    final int external = header.readUnsignedByte();
                    if (external != 0) {
                        throw new IOException("Not implemented");
                    } else {
                        if (((size - 1) & 1) != 0) {
                            throw new IOException("File names length invalid");
                        }
                        final byte[] names = new byte[(int)(size - 1)];
                        header.readFully(names);
                        int nextFile = 0;
                        int nextName = 0;
                        for (int i = 0; i < names.length; i += 2) {
                            if (names[i] == 0 && names[i+1] == 0) {
                                files[nextFile++].setName(new String(names, nextName, i-nextName, "UTF-16LE"));
                                nextName = i + 2;
                            }
                        }
                        if (nextName != names.length || nextFile != files.length) {
                            throw new IOException("Error parsing file names");
                        }
                    }
                    break;
                }
                case NID.kCTime: {
                    debug("  kCreationTime");
                    final BitSet timesDefined = readAllOrBits(header, files.length);
                    final int external = header.readUnsignedByte();
                    if (external != 0) {
                        throw new IOException("Unimplemented");
                    } else {
                        for (int i = 0; i < files.length; i++) {
                            files[i].setHasCreationDate(timesDefined.get(i));
                            if (files[i].getHasCreationDate()) {
                                files[i].setCreationDate(Long.reverseBytes(header.readLong()));
                            }
                        }
                    }
                    break;
                }
                case NID.kATime: {
                    debug("  kLastAccessTime");
                    final BitSet timesDefined = readAllOrBits(header, files.length);
                    final int external = header.readUnsignedByte();
                    if (external != 0) {
                        throw new IOException("Unimplemented");
                    } else {
                        for (int i = 0; i < files.length; i++) {
                            files[i].setHasAcessDate(timesDefined.get(i));
                            if (files[i].getHasAcessDate()) {
                                files[i].setAccessDate(Long.reverseBytes(header.readLong()));
                            }
                        }
                    }
                    break;
                }
                case NID.kMTime: {
                    debug("  kLastWriteTime");
                    final BitSet timesDefined = readAllOrBits(header, files.length);
                    final int external = header.readUnsignedByte();
                    if (external != 0) {
                        throw new IOException("Unimplemented");
                    } else {
                        for (int i = 0; i < files.length; i++) {
                            files[i].setHasLastModifiedDate(timesDefined.get(i));
                            if (files[i].getHasLastModifiedDate()) {
                                files[i].setLastModifiedDate(Long.reverseBytes(header.readLong()));
                            }
                        }
                    }
                    break;
                }
                case NID.kWinAttributes: {
                    debug("  kWinAttributes");
                    final BitSet attributesDefined = readAllOrBits(header, files.length);
                    final int external = header.readUnsignedByte();
                    if (external != 0) {
                        throw new IOException("Unimplemented");
                    } else {
                        for (int i = 0; i < files.length; i++) {
                            files[i].setHasWindowsAttributes(attributesDefined.get(i));
                            if (files[i].getHasWindowsAttributes()) {
                                files[i].setWindowsAttributes(Integer.reverseBytes(header.readInt()));
                            }
                        }
                    }
                    break;
                }
                case NID.kStartPos: {
                    debug("  kStartPos");
                    throw new IOException("kStartPos is unsupported, please report");
                }
                case NID.kDummy: {
                    debug("  kDummy");
                    throw new IOException("kDummy is unsupported, please report");
                }
                
                default: {
                    throw new IOException("Unknown property " + propertyType);
                    // FIXME: Should actually:
                    //header.skipBytes((int)size);
                }
            }
        }
        int nonEmptyFileCounter = 0;
        int emptyFileCounter = 0;
        for (int i = 0; i < files.length; i++) {
            files[i].setHasStream((isEmptyStream == null) ? true : !isEmptyStream.get(i));
            if (files[i].hasStream()) {
                files[i].setDirectory(false);
                files[i].setAntiItem(false);
                files[i].setHasCrc(archive.subStreamsInfo.hasCrc.get(nonEmptyFileCounter));
                files[i].setCrc(archive.subStreamsInfo.crcs[nonEmptyFileCounter]);
                files[i].setSize(archive.subStreamsInfo.unpackSizes[nonEmptyFileCounter]);
                ++nonEmptyFileCounter;
            } else {
                files[i].setDirectory((isEmptyFile == null) ? true : !isEmptyFile.get(emptyFileCounter));
                files[i].setAntiItem((isAnti == null) ? false : isAnti.get(emptyFileCounter));
                files[i].setHasCrc(false);
                files[i].setSize(0);
                ++emptyFileCounter;
            }
        }
        archive.files = files;
        calculateStreamMap(archive);
    }
    
    private void calculateStreamMap(final Archive archive) throws IOException {
        final StreamMap streamMap = new StreamMap();
        
        int nextFolderPackStreamIndex = 0;
        final int numFolders = (archive.folders != null) ? archive.folders.length : 0;
        streamMap.folderFirstPackStreamIndex = new int[numFolders];
        for (int i = 0; i < numFolders; i++) {
            streamMap.folderFirstPackStreamIndex[i] = nextFolderPackStreamIndex;
            nextFolderPackStreamIndex += archive.folders[i].packedStreams.length;
        }
        
        long nextPackStreamOffset = 0;
        final int numPackSizes = (archive.packSizes != null) ? archive.packSizes.length : 0;
        streamMap.packStreamOffsets = new long[numPackSizes];
        for (int i = 0; i < numPackSizes; i++) {
            streamMap.packStreamOffsets[i] = nextPackStreamOffset;
            nextPackStreamOffset += archive.packSizes[i]; 
        }
        
        streamMap.folderFirstFileIndex = new int[numFolders];
        streamMap.fileFolderIndex = new int[archive.files.length];
        int nextFolderIndex = 0;
        int nextFolderUnpackStreamIndex = 0;
        for (int i = 0; i < archive.files.length; i++) {
            if (!archive.files[i].hasStream() && nextFolderUnpackStreamIndex == 0) {
                streamMap.fileFolderIndex[i] = -1;
                continue;
            }
            if (nextFolderUnpackStreamIndex == 0) {
                for (; nextFolderIndex < archive.folders.length; ++nextFolderIndex) {
                    streamMap.folderFirstFileIndex[nextFolderIndex] = i;
                    if (archive.folders[nextFolderIndex].numUnpackSubStreams > 0) {
                        break;
                    }
                }
                if (nextFolderIndex >= archive.folders.length) {
                    throw new IOException("Too few folders in archive");
                }
            }
            streamMap.fileFolderIndex[i] = nextFolderIndex;
            if (!archive.files[i].hasStream()) {
                continue;
            }
            ++nextFolderUnpackStreamIndex;
            if (nextFolderUnpackStreamIndex >= archive.folders[nextFolderIndex].numUnpackSubStreams) {
                ++nextFolderIndex;
                nextFolderUnpackStreamIndex = 0;
            }
        }
        
        archive.streamMap = streamMap;
    }
    
    private void buildDecodingStream() throws IOException {
        final int folderIndex = archive.streamMap.fileFolderIndex[currentEntryIndex];
        if (folderIndex < 0) {
            currentEntryInputStream = new BoundedInputStream(
                    new ByteArrayInputStream(new byte[0]), 0);
            return;
        }
        if (currentFolderIndex == folderIndex) {
            // need to advance the folder input stream past the current file
            drainPreviousEntry();
        } else {
            currentFolderIndex = folderIndex;
            if (currentFolderInputStream != null) {
                currentFolderInputStream.close();
                currentFolderInputStream = null;
            }
            
            final Folder folder = archive.folders[folderIndex];
            final int firstPackStreamIndex = archive.streamMap.folderFirstPackStreamIndex[folderIndex];
            final long folderOffset = SIGNATURE_HEADER_SIZE + archive.packPos +
                    archive.streamMap.packStreamOffsets[firstPackStreamIndex];
            currentFolderInputStream = buildDecoderStack(folder, folderOffset, firstPackStreamIndex);
        }
        final SevenZArchiveEntry file = archive.files[currentEntryIndex];
        final InputStream fileStream = new BoundedInputStream(
                currentFolderInputStream, file.getSize());
        if (file.getHasCrc()) {
            currentEntryInputStream = new CRC32VerifyingInputStream(
                    fileStream, file.getSize(), file.getCrc());
        } else {
            currentEntryInputStream = fileStream;
        }
        
    }
    
    private void drainPreviousEntry() throws IOException {
        if (currentEntryInputStream != null) {
            final byte[] buffer = new byte[64*1024];
            while (currentEntryInputStream.read(buffer) >= 0) { // NOPMD
            }
            currentEntryInputStream.close();
            currentEntryInputStream = null;
        }
    }
    
    private InputStream buildDecoderStack(final Folder folder, final long folderOffset,
            final int firstPackStreamIndex) throws IOException {
        file.seek(folderOffset);
        InputStream inputStreamStack = new BoundedRandomAccessFileInputStream(file,
                archive.packSizes[firstPackStreamIndex]);
        for (final Coder coder : folder.coders) {
            if (coder.numInStreams != 1 || coder.numOutStreams != 1) {
                throw new IOException("Multi input/output stream coders are not yet supported");
            }
            inputStreamStack = Coders.addDecoder(inputStreamStack, coder, password);
        }
        if (folder.hasCrc) {
            return new CRC32VerifyingInputStream(inputStreamStack,
                    folder.getUnpackSize(), folder.crc);
        } else {
            return inputStreamStack;
        }
    }
    
    public int read() throws IOException {
        return currentEntryInputStream.read();
    }
    
    public int read(byte[] b) throws IOException {
        return read(b, 0, b.length);
    }
    
    public int read(byte[] b, int off, int len) throws IOException {
        return currentEntryInputStream.read(b, off, len);
    }
    
    private static long readUint64(final DataInput in) throws IOException {
        int firstByte = in.readUnsignedByte();
        int mask = 0x80;
        int value = 0;
        for (int i = 0; i < 8; i++) {
            if ((firstByte & mask) == 0) {
                return value | ((firstByte & (mask - 1)) << (8 * i));
            }
            int nextByte = in.readUnsignedByte();
            value |= (nextByte << (8 * i));
            mask >>>= 1;
        }
        return value;
    }
    
    private static class BoundedInputStream extends InputStream {
        private final InputStream in;
        private long bytesRemaining;
        
        public BoundedInputStream(final InputStream in, final long size) {
            this.in = in;
            bytesRemaining = size;
        }
        
        @Override
        public int read() throws IOException {
            if (bytesRemaining > 0) {
                --bytesRemaining;
                return in.read();
            } else {
                return -1;
            }
        }

        @Override
        public int read(byte[] b, int off, int len) throws IOException {
            if (bytesRemaining == 0) {
                return -1;
            }
            int bytesToRead = len;
            if (bytesToRead > bytesRemaining) {
                bytesToRead = (int) bytesRemaining;
            }
            final int bytesRead = in.read(b, off, bytesToRead);
            if (bytesRead >= 0) {
                bytesRemaining -= bytesRead;
            }
            return bytesRead;
        }

        @Override
        public void close() {
        }
    }
}
